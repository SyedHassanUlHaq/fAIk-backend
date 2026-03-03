import os
import shutil
import uuid
import json
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from validation.validate import validate_video
from config.project_config import UPLOAD_DIR, CHECKPOINT, DEVICE, THRESHOLD

router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)


class VideoRequest(BaseModel):
    video_path: str


@router.post("/upload-video")
async def upload_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
):
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail="Video file not found")
    
    if not request.video_path.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov")

    session_id = str(uuid.uuid4())

    def process_video(path: str, session_id: str):
        folder_name = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(os.path.join(folder_name, "original"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, "videos"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, "audios"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, "results"), exist_ok=True)

        try:
            result = validate_video(path, CHECKPOINT, DEVICE, THRESHOLD)
            with open(os.path.join(folder_name, "results", "probabilities.json"), "w") as f:
                json.dump({"session_id": session_id, "result": result}, f, indent=4)
        except Exception as e:
            with open(os.path.join(folder_name, "results", "error.json"), "w") as f:
                json.dump({"session_id": session_id, "error": str(e)}, f, indent=4)

    background_tasks.add_task(process_video, request.video_path, session_id)

    return {
        "message": "Video uploaded. Processing started.",
        "status": "processing",
        "session_id": session_id
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
