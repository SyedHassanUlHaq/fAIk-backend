"""
Celery task that runs the tamper / scene-cut detection AI model on a queued scan.

Flow:
  1. Download file from S3 (or a submitted URL) to a temp path.
  2. Probe duration/size with ffprobe.
  3. Update progress through the pipeline stages.
  4. Run the scene-change embedding model to find cuts/splices/re-encodes.
  5. Persist results and mark the scan complete.
"""

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

from celery.signals import worker_ready

from celery_app import celery_app
from database import SessionLocal
from models.scan import Scan
from utils.s3 import download_file, upload_file
from utils.push import send_push

_scene_model = None
_scene_processor = None
_scene_device = None


def _ensure_scene_model():
    global _scene_model, _scene_processor, _scene_device
    if _scene_model is None:
        from ml_models.scene_detection import get_embedding_model
        _scene_model, _scene_processor, _scene_device = get_embedding_model()


@worker_ready.connect
def preload_models(**kwargs):
    _ensure_scene_model()


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def _set_progress(db, scan: Scan, progress: int, stage: str):
    scan.progress = progress
    scan.current_stage = stage
    db.commit()


# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------

def _probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0))
    size = int(fmt.get("size", 0))
    bitrate = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio" and s.get("bit_rate"):
            br = int(s["bit_rate"]) // 1000
            bitrate = f"{br} kbps"
            break
    return {"duration": duration, "size": size, "bitrate": bitrate}


# ---------------------------------------------------------------------------
# Tamper / scene detection processing
# ---------------------------------------------------------------------------

def _process_tamper(scan: Scan, video_path: str, db) -> dict:
    from services.scene_detection.detector import detect_scene_changes
    from services.scene_detection.video_utils import convert_to_fps

    _set_progress(db, scan, 15, "decoding_stream")
    _ensure_scene_model()

    _set_progress(db, scan, 40, "spectral_analysis")
    converted = convert_to_fps(video_path)

    _set_progress(db, scan, 65, "frame_coherence")
    raw_cuts = detect_scene_changes(converted, _scene_model, _scene_processor, _scene_device)

    _set_progress(db, scan, 90, "cross_check_model")

    if os.path.exists(converted) and converted != video_path:
        os.remove(converted)

    # Map scene cuts to edits
    fps = 20
    edits = []
    for i, r in enumerate(raw_cuts, start=1):
        time_sec = r["frame"] / fps
        severity = "high" if r.get("emb_diff", 0) > 0.15 else "medium"
        tag = "cut"
        if r.get("mse", 0) > 6000:
            tag = "recode"
        elif r.get("emb_diff", 0) > 0.25:
            tag = "splice"
        edits.append({
            "number": i,
            "timeSeconds": round(time_sec, 1),
            "label": {"cut": "Hard cut", "splice": "Splice from another clip", "recode": "Re-encode detected"}[tag],
            "tag": tag,
            "severity": severity,
        })

    edit_count = len(edits)
    verdict = "tampered" if edit_count > 0 else "authentic"
    result_type = "editTamper" if verdict == "tampered" else "authentic"

    # Score: scale edit count — cap at 95
    score = min(95, edit_count * 20) if edit_count > 0 else 5
    tamper_level = "high" if score >= 70 else "medium" if score >= 40 else "low"

    edit_summary = (
        f"{edit_count} edit{'s' if edit_count != 1 else ''} detected. "
        + ("Likely tampered." if verdict == "tampered" else "Audio and video appear authentic.")
    )

    return {
        "verdict": verdict,
        "score": score,
        "result_type": result_type,
        "result_data": {
            "editCount": edit_count,
            "editSummary": edit_summary,
            "tamperLevel": tamper_level,
            "edits": edits,
        },
        "thumbnail_key": None,
    }


# ---------------------------------------------------------------------------
# URL download (yt-dlp)
# ---------------------------------------------------------------------------

def _download_url(url: str) -> tuple[str, str]:
    """Download a video URL using yt-dlp. Returns (local_tmp_path, detected_filename)."""
    import yt_dlp

    tmp_path = tempfile.mktemp(suffix=".mp4")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": tmp_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "url-scan") if info else "url-scan"

    actual_path = tmp_path
    if not os.path.exists(actual_path):
        for candidate in (tmp_path + ".mp4", tmp_path + ".webm"):
            if os.path.exists(candidate):
                actual_path = candidate
                break

    return actual_path, title


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="tasks.process_tamper_scan", bind=True, max_retries=2)
def process_tamper_scan(self, scan_id: str):
    db = SessionLocal()
    tmp_path = None

    try:
        scan = db.query(Scan).filter(Scan.id == uuid.UUID(scan_id)).first()
        if not scan:
            return

        scan.status = "processing"
        scan.progress = 0
        db.commit()

        if scan.url_source:
            tmp_path, detected_title = _download_url(scan.url_source)
            if not scan.filename or scan.filename == scan.url_source:
                scan.filename = detected_title
            scan.file_size = os.path.getsize(tmp_path)

            s3_key = f"scans/{scan.id}{os.path.splitext(tmp_path)[1]}"
            upload_file(tmp_path, s3_key)
            scan.file_key = s3_key
            db.commit()

        elif scan.file_key:
            tmp_path = tempfile.mktemp(suffix=".mp4")
            download_file(scan.file_key, tmp_path)

        else:
            raise RuntimeError("Scan has neither a file key nor a source URL.")

        # Probe metadata
        info = _probe(tmp_path)
        scan.duration = info["duration"]
        if not scan.file_size:
            scan.file_size = info["size"]
        if info["bitrate"] and not scan.bitrate:
            scan.bitrate = info["bitrate"]
        db.commit()

        result = _process_tamper(scan, tmp_path, db)

        scan.verdict = result["verdict"]
        scan.score = result["score"]
        scan.result_type = result["result_type"]
        scan.result_data = result["result_data"]
        scan.thumbnail_key = result.get("thumbnail_key")
        scan.status = "complete"
        scan.progress = 100
        scan.current_stage = None
        scan.completed_at = datetime.now(timezone.utc)

        scan.user.scans_used_this_month = (scan.user.scans_used_this_month or 0) + 1
        db.commit()

        verdict_label = {"tampered": "Tampered", "authentic": "Authentic"}.get(scan.verdict, "Complete")
        send_push(
            scan.user.push_token,
            title="Scan complete",
            body=f"{scan.filename} — {verdict_label} ({scan.score}/100)",
            data={"scanId": str(scan.id), "verdict": scan.verdict},
        )

    except Exception as exc:
        if db:
            try:
                scan = db.query(Scan).filter(Scan.id == uuid.UUID(scan_id)).first()
                if scan:
                    scan.status = "failed"
                    scan.error_message = str(exc)
                    db.commit()
                    send_push(
                        scan.user.push_token,
                        title="Scan failed",
                        body=f"Analysis of {scan.filename} could not be completed.",
                        data={"scanId": str(scan.id)},
                    )
            except Exception:
                pass
        raise self.retry(exc=exc, countdown=30)

    finally:
        db.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
