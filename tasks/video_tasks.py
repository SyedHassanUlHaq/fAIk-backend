import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from celery import Celery
from config.project_config import UPLOAD_DIR, THRESHOLD, DEVICE
from ml_models import video
from helpers.video_helper import split_video_into_chunks, infer_chunk

logger = logging.getLogger(__name__)

celery_app = Celery(
    "faik_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

@celery_app.task(bind=True, name="tasks.process_video")
def process_video_task(self, session_id: str, original_path: str, folder_name: str):
    try:
        self.update_state(state="PROGRESS", meta={"status": "splitting video into chunks"})

        try:
            chunks_dir = os.path.join(folder_name, "videos")
            chunks = split_video_into_chunks(original_path, chunks_dir, 5)
        except Exception as e:
            logger.error(f"Failed to split video {original_path}: {e}", exc_info=True)
            self.update_state(state="FAILURE", meta={"status": f"Video splitting failed: {str(e)}"})
            raise

        self.update_state(state="PROGRESS", meta={"status": "running inference", "total_chunks": len(chunks)})

        def run_inference(chunks):
            try:
                if DEVICE == "cuda":
                    return [infer_chunk(c) for c in chunks]
                with ThreadPoolExecutor(max_workers=min(len(chunks), os.cpu_count() or 4)) as ex:
                    return list(ex.map(infer_chunk, chunks))
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                raise

        try:
            chunk_results = run_inference(chunks)
        except Exception as e:
            logger.error(f"Failed to run inference: {e}", exc_info=True)
            self.update_state(state="FAILURE", meta={"status": f"Inference failed: {str(e)}"})
            raise

        try:
            total_prob = sum(r["result"].get("probability", 0.0) for r in chunk_results)

            overall_prob = total_prob / len(chunks) if chunks else 0.0
            overall_prediction = "Fake" if overall_prob >= THRESHOLD else "Real"

            result_payload = {
                "session_id": session_id,
                "chunks": chunk_results,
                "overall": {
                    "probability": overall_prob,
                    "prediction": overall_prediction,
                    "threshold": THRESHOLD,
                },
            }

            results_path = os.path.join(folder_name, "results", "probabilities.json")
            with open(results_path, "w") as f:
                json.dump(result_payload, f, indent=4)

            return result_payload
        except IOError as e:
            logger.error(f"Failed to write results to {results_path}: {e}", exc_info=True)
            self.update_state(state="FAILURE", meta={"status": f"Failed to save results: {str(e)}"})
            raise
        except Exception as e:
            logger.error(f"Error processing results: {e}", exc_info=True)
            self.update_state(state="FAILURE", meta={"status": f"Result processing failed: {str(e)}"})
            raise

    except Exception as exc:
        logger.error(f"Task failed for session {session_id}: {exc}", exc_info=True)
        self.update_state(state="FAILURE", meta={"status": str(exc)})
        raise