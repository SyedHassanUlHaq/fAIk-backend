import asyncio
import os
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from repositories.validation_tool import validate_video
from config.project_config import UPLOAD_DIR, THRESHOLD, DEVICE
from ml_models import video
from helpers.video_helper import split_video_into_chunks, infer_chunk
from datetime import datetime

router = APIRouter()

class VideoRequest(BaseModel):
    video_path: str


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    print(f"[INFO] {datetime.now()}: Received video upload request: {file.filename}")

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    session_id = str(uuid.uuid4())
    folder_name = os.path.join(UPLOAD_DIR, session_id)

    dirs = ["original", "videos", "audios", "results"]
    for d in dirs:
        os.makedirs(os.path.join(folder_name, d), exist_ok=True)

    loop = asyncio.get_event_loop()

    try:
        original_path = os.path.join(folder_name, "original", file.filename)

        # Chunked upload: avoids loading the entire file into memory at once
        with open(original_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        print(f"[INFO] Saved uploaded file at: {original_path}")

        chunks_dir = os.path.join(folder_name, "videos")

        chunks = await loop.run_in_executor(
            None, split_video_into_chunks, original_path, chunks_dir, 5
        )

        def run_inference(chunks):
            if DEVICE == "cuda":
                return [infer_chunk(c) for c in chunks]
            with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 4)) as ex:
                return list(ex.map(infer_chunk, chunks))

        chunk_results = await loop.run_in_executor(None, run_inference, chunks)
        total_prob = sum(r["result"].get("probability", 0.0) for r in chunk_results)

        overall_prob = total_prob / len(chunks) if chunks else 0.0
        overall_prediction = "Fake" if overall_prob >= THRESHOLD else "Real"

        results_path = os.path.join(folder_name, "results", "probabilities.json")

        with open(results_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "chunks": chunk_results,
                "overall": {
                    "probability": overall_prob,
                    "prediction": overall_prediction,
                    "threshold": THRESHOLD
                }
            }, f, indent=4)

        return {
            "session_id": session_id,
            "total_chunks": len(chunk_results),
            "results": chunk_results,
            "overall": {
                "probability": overall_prob,
                "prediction": overall_prediction,
                "threshold": THRESHOLD
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{session_id}")
def get_results(session_id: str):
    result_path = os.path.join(UPLOAD_DIR, session_id, "results", "probabilities.json")

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