from ml_models.scene_detection import get_embedding_model
from services.scene_detection.detector import detect_scene_changes
from services.scene_detection.video_utils import convert_to_fps
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
import asyncio
import functools
import os
import tempfile
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

#----------------------------------------------------------------------------------------
# Scene Detection Utilities
#----------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------
# Scene Detection Endpoints
#----------------------------------------------------------------------------------------

@router.post("/detect-scenes")
async def detect_scenes(request: Request, video: UploadFile = File(...)):
    start_time = time.time()
    tmp_path = None
    
    try:
        print("\n🚀 [START] Scene detection request received")
        print(f"📁 Filename: {video.filename}")

        try:
            model = request.app.state.embedding_model
            processor = request.app.state.embedding_processor
            device = request.app.state.embedding_device
        except AttributeError as e:
            logger.error(f"Embedding model not loaded: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="Scene detection model not initialized")

        if model is None or processor is None:
            logger.error("Embedding models are None")
            raise HTTPException(status_code=503, detail="Scene detection model not available")

        print(f"🧠 Model device: {device}")

        upload_start = time.time()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                total_size = 0
                while chunk := await video.read(1024 * 1024): 
                    tmp_file.write(chunk)
                    total_size += len(chunk)
                tmp_path = tmp_file.name
        except Exception as e:
            logger.error(f"Failed to save uploaded video: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save uploaded video")

        print(f"📥 Upload saved to temp file: {tmp_path}")
        print(f"📦 File size: {round(total_size / (1024*1024), 2)} MB")
        print(f"⏱️ Upload time: {round(time.time() - upload_start, 2)} sec")

        loop = asyncio.get_event_loop()

        try:
            convert_start = time.time()
            print("Converting video to 20 FPS...")
            # Offload blocking CPU work so the event loop stays free for other requests
            try:
                converted = await loop.run_in_executor(None, convert_to_fps, tmp_path)
            except Exception as e:
                logger.error(f"Failed to convert video: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to convert video format")
            
            print(f"Converted video path: {converted}")
            print(f"Conversion time: {round(time.time() - convert_start, 2)} sec")

            detect_start = time.time()
            print("Running scene detection...")
            try:
                raw_results = await loop.run_in_executor(
                    None,
                    functools.partial(detect_scene_changes, converted, model, processor, device)
                )
            except Exception as e:
                logger.error(f"Scene detection failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Scene detection processing failed")
            
            print(f"Detection complete. Found {len(raw_results)} cuts")
            print(f"Detection time: {round(time.time() - detect_start, 2)} sec")

            fps = 20
            formatted_results = []

            try:
                for i, r in enumerate(raw_results, start=1):
                    try:
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
                    except Exception as e:
                        logger.warning(f"Error formatting result {i}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error processing raw results: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to process detection results")

            total_time = round(time.time() - start_time, 2)

            print(f"🎉 [DONE] Total processing time: {total_time} sec\n")

            return {
                "video_name": video.filename,
                "file_size_mb": round(total_size / (1024*1024), 2),
                "processing_time_sec": total_time,
                "total_scenes_detected": len(formatted_results),
                "scenes": formatted_results
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Scene detection processing failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in detect_scenes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scene detection request failed")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🧹 Temp file deleted: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")