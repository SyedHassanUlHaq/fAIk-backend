import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict

# Import your existing functions from main.py
from helpers.helpers import run_fake_detection_session

app = FastAPI(title="Deepfake Detection API")

# Ensure the base sessions directory exists
UPLOAD_DIR = "sessions"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dictionary to keep track of job status (In-memory for this example)
jobs = {}

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a video and starts the background processing.
    Returns a session_id to track progress.
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    # 1. Create a temporary landing spot for the raw upload
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Assign job status
    # We'll let your run_fake_detection_session handle the UUID generation internally
    # but we trigger it as a background task.
    
    def process_video_task(path: str):
        try:
            results = run_fake_detection_session(path)
            # Cleanup the temporary uploaded file after processing
            if os.path.exists(path):
                os.remove(path)
            return results
        except Exception as e:
            print(f"Error processing session: {e}")

    background_tasks.add_task(process_video_task, temp_path)

    return {
        "message": "Video uploaded successfully. Processing started in background.",
        "temp_file": temp_path
    }

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """
    Retrieve the results of a session by reading the generated JSON file.
    """
    result_path = f"sessions/{session_id}/results/probabilities.json"
    
    if not os.path.exists(result_path):
        return JSONResponse(
            status_code=404, 
            content={"message": "Results not found. The session might still be processing or the ID is invalid."}
        )
    
    import json
    with open(result_path, "r") as f:
        data = json.load(f)
    
    return data

@app.get("/health")
def health_check():
    return {"status": "active"}