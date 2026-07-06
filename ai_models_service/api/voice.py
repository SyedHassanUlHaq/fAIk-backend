from fastapi import APIRouter

router = APIRouter()

# Voice / AI-audio detection endpoints — model not yet integrated.
# The /v1/scans pipeline in core_service already routes scan_type="audio"
# to tasks.process_audio_scan (see tasks/audio_scan_task.py), which currently
# returns a stub "authentic" verdict until a voice model lands here.


@router.get("/health")
def health_check():
    return {"status": "active"}
