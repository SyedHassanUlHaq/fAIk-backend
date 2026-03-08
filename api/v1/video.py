import os
import shutil
import uuid
import json
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from validation.validate import validate_video
from config.project_config import UPLOAD_DIR, CHECKPOINT, DEVICE, THRESHOLD
from helpers.video_helper import split_video_into_intervals

router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)


class VideoRequest(BaseModel):
    video_path: str


@router.post("/upload-video")
async def upload_video(request: VideoRequest):

    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail="Video file not found")

    if not request.video_path.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported: .mp4, .avi, .mov"
        )

    session_id = str(uuid.uuid4())

    folder_name = os.path.join(UPLOAD_DIR, session_id)

    os.makedirs(os.path.join(folder_name, "original"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "videos"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "audios"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "results"), exist_ok=True)

    try:
        # 1️⃣ Split video into 5 second chunks
        chunks = split_video_into_intervals(request.video_path, folder_name)

        chunk_results = []

        # 2️⃣ Run inference on each chunk
        for chunk_path in chunks:
            result = validate_video(chunk_path, CHECKPOINT, DEVICE, THRESHOLD)

            chunk_results.append({
                "chunk": os.path.basename(chunk_path),
                "path": chunk_path,
                "result": result
            })

        # 3️⃣ Save results
        results_path = os.path.join(folder_name, "results", "probabilities.json")

        with open(results_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "chunks": chunk_results
            }, f, indent=4)

        # 4️⃣ Return results
        return {
            "session_id": session_id,
            "total_chunks": len(chunk_results),
            "results": chunk_results
        }

    except Exception as e:

        error_path = os.path.join(folder_name, "results", "error.json")

        with open(error_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "error": str(e)
            }, f, indent=4)

        raise HTTPException(status_code=500, detail=str(e))

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
