"""
Celery task that runs AI-voice / audio-tamper detection on a queued scan.

The voice model isn't integrated yet, so this currently downloads/probes the
file and returns a stub "authentic" verdict — the same placeholder behavior
the monolith's scan pipeline used before the AI-model split. Once a voice
model lands, plug its inference into `_process_audio` below.
"""

import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

from celery_app import celery_app
from database import SessionLocal
from models.scan import Scan
from utils.s3 import download_file, upload_file
from utils.push import send_push


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
# Audio stub (voice model not yet integrated)
# ---------------------------------------------------------------------------

def _process_audio(scan: Scan, audio_path: str, db) -> dict:
    _set_progress(db, scan, 90, "cross_check_model")
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

def _download_url(url: str) -> tuple[str, str]:
    """Download an audio URL using yt-dlp. Returns (local_tmp_path, detected_filename)."""
    import yt_dlp

    tmp_path = tempfile.mktemp(suffix=".m4a")
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": tmp_path,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "url-scan") if info else "url-scan"

    actual_path = tmp_path
    if not os.path.exists(actual_path):
        for candidate in (tmp_path + ".m4a", tmp_path + ".webm"):
            if os.path.exists(candidate):
                actual_path = candidate
                break

    return actual_path, title


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="tasks.process_audio_scan", bind=True, max_retries=2)
def process_audio_scan(self, scan_id: str):
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
            tmp_path = tempfile.mktemp(suffix=".m4a")
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

        scan.user.scans_used_this_month = (scan.user.scans_used_this_month or 0) + 1
        db.commit()

        send_push(
            scan.user.push_token,
            title="Scan complete",
            body=f"{scan.filename} — Authentic ({scan.score}/100)",
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
