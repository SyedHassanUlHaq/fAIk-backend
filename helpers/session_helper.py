# Handles orchestration of the session (combining everything).
import os
import uuid
import json
from typing import Dict

from helpers.audio_helper import extract_audio_from_video, split_audio_into_intervals
from helpers.video_helper import split_video_into_intervals
from helpers.prediction_helper import predict_all_video_intervals, predict_all_audio_intervals, combine_interval_probabilities

def run_fake_detection_session(video_path: str) -> Dict:
    """End-to-end orchestration for frontend and API usage."""
    
    session_id = str(uuid.uuid4())
    folder_name = f"sessions/{session_id}"
    
    os.makedirs(f"{folder_name}/original", exist_ok=True)
    os.makedirs(f"{folder_name}/videos", exist_ok=True)
    os.makedirs(f"{folder_name}/audios", exist_ok=True)
    os.makedirs(f"{folder_name}/results", exist_ok=True)

    original_audio_path = extract_audio_from_video(video_path, folder_name)
    
    split_video_into_intervals(video_path, folder_name)
    split_audio_into_intervals(original_audio_path, folder_name)
    
    video_probs = predict_all_video_intervals(folder_name)
    audio_probs = predict_all_audio_intervals(folder_name)
    
    combined_results = combine_interval_probabilities(video_probs, audio_probs)
    
    final_response = {
        "session_id": session_id,
        "intervals": combined_results
    }

    with open(f"{folder_name}/results/probabilities.json", "w") as f:
        json.dump(final_response, f, indent=4)
        
    return final_response
