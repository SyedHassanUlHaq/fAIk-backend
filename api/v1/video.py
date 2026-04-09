import os
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from repositories.validation_tool import validate_video
from config.project_config import UPLOAD_DIR
from tasks.video_tasks import process_video_task

router = APIRouter()

#----------------------------------------------------------------------------------------
# Video Analysis Schemas
#----------------------------------------------------------------------------------------

class VideoRequest(BaseModel):
    video_path: str

#----------------------------------------------------------------------------------------
# Video Analysis Endpoints
#----------------------------------------------------------------------------------------

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    print(f"[INFO] {datetime.now()}: Received upload: {file.filename}")

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    session_id = str(uuid.uuid4())
    folder_name = os.path.join(UPLOAD_DIR, session_id)

    for d in ["original", "videos", "audios", "results"]:
        os.makedirs(os.path.join(folder_name, d), exist_ok=True)

    original_path = os.path.join(folder_name, "original", file.filename)

    # Save file (chunked, non-blocking)
    try:
        with open(original_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Enqueue heavy work — returns immediately
    task = process_video_task.delay(session_id, original_path, folder_name)

    return {
        "session_id": session_id,
        "task_id": task.id,
        "status": "queued",
        "message": "Video uploaded. Poll /results/{session_id} for results.",
    }


@router.get("/status/{task_id}")
def get_task_status(task_id: str):
    """Optional: check raw Celery task state."""
    from tasks.video_tasks import celery_app
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,         # PENDING / PROGRESS / SUCCESS / FAILURE
        "info": result.info,
    }


@router.get("/results/{session_id}")
def get_results(session_id: str):
    result_path = os.path.join(UPLOAD_DIR, session_id, "results", "probabilities.json")

    if not os.path.exists(result_path):
        return JSONResponse(
            status_code=202,
            content={"message": "Processing in progress or invalid session_id", "session_id": session_id},
        )

    with open(result_path, "r") as f:
        return json.load(f)


@router.get("/health")
def health_check():
    return {"status": "active"}