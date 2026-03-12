import os
import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from validation.validate import validate_video
from config.project_config import UPLOAD_DIR, CHECKPOINT, DEVICE, THRESHOLD
from ml_models.loader import raft_model, fused_model
from helpers.video_helper import split_video_into_chunks

router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)


class VideoRequest(BaseModel):
    video_path: str


# ----------------------------
# Top-level inference function
# ----------------------------
def infer_chunk(chunk_path):
    """
    Run validation on a single video chunk.
    Uses globally loaded raft_model and fused_model.
    """
    try:
        print(f"[INFO] Starting inference for chunk: {chunk_path}")
        result = validate_video(chunk_path, raft_model, fused_model, DEVICE, THRESHOLD)
        print(f"[INFO] Completed inference for chunk: {chunk_path}, probability: {result['probability']:.4f}")
        return {
            "chunk": os.path.basename(chunk_path),
            "path": chunk_path,
            "result": result
        }
    except Exception as e:
        print(f"[ERROR] Chunk inference failed: {chunk_path}, error: {e}")
        return {
            "chunk": os.path.basename(chunk_path),
            "path": chunk_path,
            "result": {"probability": 0.0, "error": str(e)}
        }


# ----------------------------
# Upload-video route
# ----------------------------
@router.post("/upload-video")
async def upload_video(request: VideoRequest):
    print(f"[INFO] Received video upload request: {request.video_path}")

    if not os.path.exists(request.video_path):
        print(f"[ERROR] Video file not found: {request.video_path}")
        raise HTTPException(status_code=400, detail="Video file not found")

    if not request.video_path.endswith((".mp4", ".avi", ".mov")):
        print(f"[ERROR] Invalid video format: {request.video_path}")
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov")

    session_id = str(uuid.uuid4())
    folder_name = os.path.join(UPLOAD_DIR, session_id)
    print(f"[INFO] Starting session: {session_id}")

    # Create directories
    dirs = ["original", "videos", "audios", "results"]
    for d in dirs:
        path = os.path.join(folder_name, d)
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")

    try:
        # Split video into 5-second chunks
        chunks_dir = os.path.join(folder_name, "videos")
        print(f"[INFO] Splitting video into chunks at: {chunks_dir}")
        chunks = split_video_into_chunks(request.video_path, chunks_dir, chunk_length=5)
        print(f"[INFO] Total chunks created: {len(chunks)}")

        chunk_results = []
        total_prob = 0.0

        # Run inference in parallel using threads
        print(f"[INFO] Running inference in parallel for all chunks...")
        max_workers = min(len(chunks), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(infer_chunk, c) for c in chunks]
            for future in as_completed(futures):
                res = future.result()
                chunk_results.append(res)
                total_prob += res["result"].get("probability", 0.0)
                print(f"[INFO] Aggregated result for chunk: {res['chunk']}")

        # Compute overall probability
        overall_prob = total_prob / len(chunks) if chunks else 0.0
        overall_prediction = "Fake" if overall_prob >= THRESHOLD else "Real"
        print(f"[INFO] Overall probability: {overall_prob:.4f}, prediction: {overall_prediction}")

        # Save results
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
        print(f"[INFO] Results saved at: {results_path}")

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
        error_path = os.path.join(folder_name, "results", "error.json")
        with open(error_path, "w") as f:
            json.dump({"session_id": session_id, "error": str(e)}, f, indent=4)
        print(f"[ERROR] Exception occurred: {e}. Error saved at: {error_path}")
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