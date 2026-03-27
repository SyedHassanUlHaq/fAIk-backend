from ml_models.scene_detection import get_embedding_model
from services.scene_detection.detector import detect_scene_changes
from services.scene_detection.video_utils import convert_to_fps
from fastapi import APIRouter, File, UploadFile, Request
import asyncio
import functools
import os
import tempfile
import time

router = APIRouter()


def format_timestamp(seconds: float):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def get_strength(val, type_):
    if type_ == "ssim":
        return "high" if val < 0.5 else "moderate"
    if type_ == "mse":
        return "high" if val > 4000 else "moderate"
    if type_ == "emb":
        return "high" if val > 0.2 else "moderate"


@router.post("/detect-scenes")
async def detect_scenes(request: Request, video: UploadFile = File(...)):
    start_time = time.time()

    print("\n🚀 [START] Scene detection request received")
    print(f"📁 Filename: {video.filename}")

    model = request.app.state.embedding_model
    processor = request.app.state.embedding_processor
    device = request.app.state.embedding_device

    print(f"🧠 Model device: {device}")

    upload_start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        total_size = 0
        while chunk := await video.read(1024 * 1024): 
            tmp_file.write(chunk)
            total_size += len(chunk)
        tmp_path = tmp_file.name

    print(f"📥 Upload saved to temp file: {tmp_path}")
    print(f"📦 File size: {round(total_size / (1024*1024), 2)} MB")
    print(f"⏱️ Upload time: {round(time.time() - upload_start, 2)} sec")

    loop = asyncio.get_event_loop()

    try:
        convert_start = time.time()
        print("Converting video to 20 FPS...")
        # Offload blocking CPU work so the event loop stays free for other requests
        converted = await loop.run_in_executor(None, convert_to_fps, tmp_path)
        print(f"Converted video path: {converted}")
        print(f"Conversion time: {round(time.time() - convert_start, 2)} sec")

        detect_start = time.time()
        print("Running scene detection...")
        raw_results = await loop.run_in_executor(
            None,
            functools.partial(detect_scene_changes, converted, model, processor, device)
        )
        print(f"Detection complete. Found {len(raw_results)} cuts")
        print(f"Detection time: {round(time.time() - detect_start, 2)} sec")

        fps = 20
        formatted_results = []

        for i, r in enumerate(raw_results, start=1):
            time_sec = r["frame"] / fps

            formatted_results.append({
                "scene_number": i,
                "frame": r["frame"],
                "timestamp": format_timestamp(time_sec),
                "time_seconds": round(time_sec, 2),
                "confidence": "high" if r["emb_diff"] > 0.15 else "medium",
                "change_strength": {
                    "visual_change (ssim)": get_strength(r["ssim"], "ssim"),
                    "pixel_change (mse)": get_strength(r["mse"], "mse"),
                    "semantic_change (ai)": get_strength(r["emb_diff"], "emb")
                },
                "raw_metrics": {
                    "ssim": round(r["ssim"], 4),
                    "mse": round(r["mse"], 2),
                    "emb_diff": round(r["emb_diff"], 4)
                }
            })

        total_time = round(time.time() - start_time, 2)

        print(f"🎉 [DONE] Total processing time: {total_time} sec\n")

        return {
            "video_name": video.filename,
            "file_size_mb": round(total_size / (1024*1024), 2),
            "processing_time_sec": total_time,
            "total_scenes_detected": len(formatted_results),
            "scenes": formatted_results
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"🧹 Temp file deleted: {tmp_path}")