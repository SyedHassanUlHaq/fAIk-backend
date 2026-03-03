import os
import shutil
import uuid
import json
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from helpers.session_helper import run_fake_detection_session
from config.project_config import UPLOAD_DIR

router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)


@router.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    # unique temp file
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("temp", temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def process_video(path: str):
        try:
            run_fake_detection_session(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    background_tasks.add_task(process_video, temp_path)

    return {
        "message": "Video uploaded. Processing started.",
        "status": "processing"
    }


@router.get("/results/{session_id}")
def get_results(session_id: str):
    result_path = f"sessions/{session_id}/results/probabilities.json"

    if not os.path.exists(result_path):
        return JSONResponse(
            status_code=404,
            content={"message": "Results not ready or invalid session_id"}
        )

    with open(result_path, "r") as f:
        return json.load(f)


@router.get("/health")
def health_check():
    return {"status": "active"}
