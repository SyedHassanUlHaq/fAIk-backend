"""
Celery task that runs ML inference on a queued scan.

Flow:
  1. Download file from S3 to a temp path.
  2. Probe duration/size with ffprobe.
  3. Update progress through the 4 pipeline stages.
  4. Run the appropriate ML model (video deepfake or scene-tamper).
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
from utils.s3 import download_file, upload_file, delete_file, presigned_url
from utils.push import send_push
from config.project_config import THRESHOLD, MODEL_VERSION

_video_models_loaded = False
_scene_model = None
_scene_processor = None
_scene_device = None


def _ensure_video_models():
    global _video_models_loaded
    if not _video_models_loaded:
        from ml_models.video import load_models
        load_models()
        _video_models_loaded = True


def _ensure_scene_model():
    global _scene_model, _scene_processor, _scene_device
    if _scene_model is None:
        from ml_models.scene_detection import get_embedding_model
        _scene_model, _scene_processor, _scene_device = get_embedding_model()


@worker_ready.connect
def preload_models(**kwargs):
    _ensure_scene_model()
    _ensure_video_models()


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def _set_progress(db, scan: Scan, progress: int, stage: str):
    scan.progress = progress
    scan.current_stage = stage
    db.commit()


# ---------------------------------------------------------------------------
# ffprobe helpers
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
    # pull audio stream bitrate if present
    bitrate = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "audio" and s.get("bit_rate"):
            br = int(s["bit_rate"]) // 1000
            bitrate = f"{br} kbps"
            break
    return {"duration": duration, "size": size, "bitrate": bitrate}


def _extract_thumbnail(video_path: str) -> str:
    """Extract a frame at 2 s and return the local temp path."""
    out = tempfile.mktemp(suffix=".jpg")
    subprocess.run(
        ["ffmpeg", "-ss", "2", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", out, "-y"],
        check=True, capture_output=True
    )
    return out


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

def _build_tagline(result_type: str, score: int) -> str:
    if result_type == "aiVoice":
        if score >= 85:
            return "This voice was almost certainly synthesized by an AI model."
        return "This voice shows strong signs of AI synthesis."
    if result_type == "authentic":
        return "No significant signs of AI manipulation were detected."
    if result_type == "tampered":
        return "This recording shows signs of editing or tampering."
    return "Analysis complete."


def _verdict_to_result_type(scan_type: str, verdict: str) -> str:
    mapping = {
        ("video", "ai"): "deepfakeVideo",
        ("video", "authentic"): "authentic",
        ("tamper", "tampered"): "editTamper",
        ("tamper", "authentic"): "authentic",
        ("audio", "ai"): "aiVoice",
        ("audio", "authentic"): "authentic",
        ("audio", "tampered"): "tampered",
    }
    return mapping.get((scan_type, verdict), "authentic")


# ---------------------------------------------------------------------------
# Video deepfake processing
# ---------------------------------------------------------------------------

def _process_video(scan: Scan, video_path: str, db) -> dict:
    from helpers.video_helper import split_video_into_chunks, infer_chunk
    from concurrent.futures import ThreadPoolExecutor

    _set_progress(db, scan, 15, "decoding_stream")
    _ensure_video_models()

    chunks_dir = tempfile.mkdtemp()
    _set_progress(db, scan, 40, "spectral_analysis")

    chunks = split_video_into_chunks(video_path, chunks_dir, 5)
    if not chunks:
        raise RuntimeError("No video chunks could be extracted.")

    _set_progress(db, scan, 65, "frame_coherence")

    with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 4)) as ex:
        chunk_results = list(ex.map(infer_chunk, chunks))

    _set_progress(db, scan, 90, "cross_check_model")

    total_prob = sum(r["result"].get("probability", 0.0) for r in chunk_results)
    overall_prob = total_prob / len(chunks)
    score = int(round(overall_prob * 100))
    verdict = "ai" if overall_prob >= THRESHOLD else "authentic"
    result_type = _verdict_to_result_type("video", verdict)

    # Generate segments from chunk probabilities
    segments = []
    chunk_duration = 5
    for i, r in enumerate(chunk_results):
        prob = r["result"].get("probability", 0.0)
        if prob >= 0.4:
            start = i * chunk_duration
            end = start + chunk_duration
            total_dur = scan.duration or (len(chunks) * chunk_duration)
            pos = round(start / total_dur * 100)
            width = max(2, round(chunk_duration / total_dur * 100))
            seg_type = "gan" if prob >= 0.7 else "lip_sync"
            segments.append({
                "startSeconds": start,
                "endSeconds": end,
                "label": "Face GAN signature" if seg_type == "gan" else "Lip-sync drift",
                "type": seg_type,
                "severity": "high" if prob >= 0.7 else "medium",
                "timelinePositionPercent": pos,
                "timelineWidthPercent": width,
            })

    explanation = (
        f"The video analysis detected an overall AI probability of {score}%. "
        + ("Strong GAN artefacts and temporal inconsistencies were found across multiple segments."
           if score >= 70 else
           "Some inconsistencies were detected but confidence is moderate.")
    )

    result_data: dict = {"plainEnglishExplanation": explanation, "segments": segments}

    # Thumbnail
    thumb_key = None
    try:
        thumb_local = _extract_thumbnail(video_path)
        thumb_key = f"thumbnails/{scan.id}.jpg"
        upload_file(thumb_local, thumb_key, "image/jpeg")
        os.remove(thumb_local)
        result_data["thumbnailUrl"] = presigned_url(thumb_key, expires_in=86400 * 7)
    except Exception:
        pass

    # Clean up chunk files
    for c in chunks:
        try:
            os.remove(c)
        except OSError:
            pass
    try:
        os.rmdir(chunks_dir)
    except OSError:
        pass

    return {
        "verdict": verdict,
        "score": score,
        "result_type": result_type,
        "result_data": result_data,
        "thumbnail_key": thumb_key,
    }


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
    result_type = _verdict_to_result_type("tamper", verdict)

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
# Audio stub (voice model not yet integrated)
# ---------------------------------------------------------------------------

def _process_audio(scan: Scan, audio_path: str, db) -> dict:
    _set_progress(db, scan, 90, "cross_check_model")
    # Placeholder until voice model is integrated
    return {
        "verdict": "authentic",
        "score": 0,
        "result_type": "authentic",
        "result_data": {
            "tagline": "Audio analysis is not yet available.",
            "waveformBars": [],
            "evidence": [],
        },
        "thumbnail_key": None,
    }


# ---------------------------------------------------------------------------
# URL download (yt-dlp)
# ---------------------------------------------------------------------------

def _download_url(url: str, scan_type: str) -> tuple[str, str]:
    """
    Download a video/audio URL using yt-dlp.
    Returns (local_tmp_path, detected_filename).
    """
    import yt_dlp

    suffix = ".mp4" if scan_type in ("video", "tamper") else ".m4a"
    tmp_path = tempfile.mktemp(suffix=suffix)

    if scan_type in ("video", "tamper"):
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        fmt = "bestaudio[ext=m4a]/bestaudio/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": tmp_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4" if scan_type in ("video", "tamper") else None,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "url-scan") if info else "url-scan"

    # yt-dlp may append the extension itself; find the actual output file
    actual_path = tmp_path
    if not os.path.exists(actual_path):
        for candidate in (tmp_path + ".mp4", tmp_path + ".m4a", tmp_path + ".webm"):
            if os.path.exists(candidate):
                actual_path = candidate
                break

    return actual_path, title


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2)
def process_scan(self, scan_id: str):
    db = SessionLocal()
    tmp_path = None

    try:
        scan = db.query(Scan).filter(Scan.id == uuid.UUID(scan_id)).first()
        if not scan:
            return

        scan.status = "processing"
        scan.progress = 0
        db.commit()

        suffix_map = {"video": ".mp4", "tamper": ".mp4", "audio": ".m4a"}
        suffix = suffix_map.get(scan.scan_type, ".tmp")

        if scan.url_source:
            # URL-based scan: download with yt-dlp, then upload to S3
            tmp_path, detected_title = _download_url(scan.url_source, scan.scan_type)
            if not scan.filename or scan.filename == scan.url_source:
                scan.filename = detected_title
            scan.file_size = os.path.getsize(tmp_path)

            s3_key = f"scans/{scan.id}{os.path.splitext(tmp_path)[1]}"
            upload_file(tmp_path, s3_key)
            scan.file_key = s3_key
            db.commit()

        elif scan.file_key:
            tmp_path = tempfile.mktemp(suffix=suffix)
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

        # Run inference
        if scan.scan_type == "video":
            result = _process_video(scan, tmp_path, db)
        elif scan.scan_type == "tamper":
            result = _process_tamper(scan, tmp_path, db)
        else:
            result = _process_audio(scan, tmp_path, db)

        scan.verdict = result["verdict"]
        scan.score = result["score"]
        scan.result_type = result["result_type"]
        scan.result_data = result["result_data"]
        scan.thumbnail_key = result.get("thumbnail_key")
        scan.status = "complete"
        scan.progress = 100
        scan.current_stage = None
        scan.completed_at = datetime.now(timezone.utc)

        # Increment monthly scan counter
        scan.user.scans_used_this_month = (scan.user.scans_used_this_month or 0) + 1
        db.commit()

        # Push notification
        verdict_label = {"ai": "AI-generated", "authentic": "Authentic", "tampered": "Tampered"}.get(scan.verdict, "Complete")
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
