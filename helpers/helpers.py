import os
import uuid
import json
from typing import List, Dict
# Updated for MoviePy 2.0+ compatibility
from moviepy import VideoFileClip, AudioFileClip 
from external_models import predict_video, predict_audio

# --- 7. run_fake_detection_session (Outer Wrapper) ---
def run_fake_detection_session(video_path: str) -> Dict:
    """End-to-end orchestration for frontend and API usage[cite: 124]."""
    
    # 1. Generate session_id using UUID [cite: 128]
    session_id = str(uuid.uuid4())
    folder_name = f"sessions/{session_id}"
    
    # 2. Create session folder structure [cite: 129]
    os.makedirs(f"{folder_name}/original", exist_ok=True)
    os.makedirs(f"{folder_name}/videos", exist_ok=True)
    os.makedirs(f"{folder_name}/audios", exist_ok=True)
    os.makedirs(f"{folder_name}/results", exist_ok=True)

    # 3. Save original video reference [cite: 130]
    # (In this local script, we use the video_path provided)
    
    # 4. Extract audio from video [cite: 131]
    original_audio_path = extract_audio_from_video(video_path, folder_name)
    
    # 5 & 6. Split media into 5-second intervals [cite: 132, 133]
    split_video_into_intervals(video_path, folder_name)
    split_audio_into_intervals(original_audio_path, folder_name)
    
    # 7 & 8. Predict probabilities [cite: 135, 136]
    video_probs = predict_all_video_intervals(folder_name)
    audio_probs = predict_all_audio_intervals(folder_name)
    
    # 9. Combine probabilities per interval [cite: 137]
    combined_results = combine_interval_probabilities(video_probs, audio_probs)
    
    # 10. Return structured response [cite: 138]
    final_response = {
        "session_id": session_id,
        "intervals": combined_results
    }
    
    # Save to results/probabilities.json [cite: 21]
    with open(f"{folder_name}/results/probabilities.json", "w") as f:
        json.dump(final_response, f, indent=4)
        
    return final_response

# --- 5. extract_audio_from_video ---
def extract_audio_from_video(video_path: str, folder_name: str) -> str:
    """Extracts full audio track from video[cite: 98]."""
    output_path = f"{folder_name}/audios/original_audio.wav"
    with VideoFileClip(video_path) as video:
        if video.audio is not None:
            video.audio.write_audiofile(output_path, logger=None)
    return output_path

# --- 1 & 2. Splitting Functions ---
def split_video_into_intervals(video_path: str, folder_name: str) -> List[str]:
    """Splits a video into 5-second chunks[cite: 25]."""
    paths = []
    with VideoFileClip(video_path) as clip:
        duration = int(clip.duration)
        for start in range(0, duration, 5):
            end = min(start + 5, duration)
            target = f"{folder_name}/videos/video_{start:03d}.mp4"
            subclip = clip.subclipped(start, end) # Updated for MoviePy 2.0
            subclip.write_videofile(target, codec="libx264", audio_codec="aac", logger=None)
            paths.append(target)
    return paths

def split_audio_into_intervals(audio_path: str, folder_name: str) -> List[str]:
    """Splits an audio file into 5-second chunks[cite: 38]."""
    paths = []
    if not os.path.exists(audio_path): return paths
    
    with AudioFileClip(audio_path) as audio:
        duration = int(audio.duration)
        for start in range(0, duration, 5):
            end = min(start + 5, duration)
            target = f"{folder_name}/audios/audio_{start:03d}.wav"
            subclip = audio.subclipped(start, end) # Updated for MoviePy 2.0
            subclip.write_audiofile(target, logger=None)
            paths.append(target)
    return paths

# --- 3 & 4. Prediction Wrappers ---
def predict_all_video_intervals(folder_path: str) -> Dict:
    """Runs the video model on all video intervals[cite: 52]."""
    results = []
    video_dir = f"{folder_path}/videos/"
    files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    for i, filename in enumerate(files):
        # Call predict_video(video_path) [cite: 57]
        prob = predict_video(os.path.join(video_dir, filename))
        results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
    return {"video": results}

def predict_all_audio_intervals(folder_path: str) -> Dict:
    """Runs the audio model on all audio intervals[cite: 71]."""
    results = []
    audio_dir = f"{folder_path}/audios/"
    # Exclude the original extracted file from interval predictions
    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav") and "original" not in f])
    for i, filename in enumerate(files):
        # Call predict_audio(audio_path) [cite: 77]
        prob = predict_audio(os.path.join(audio_dir, filename))
        results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
    return {"audio": results}

# --- 6. combine_interval_probabilities ---
def combine_interval_probabilities(video_probs: Dict, audio_probs: Dict) -> List[Dict]:
    """Combines audio and video probabilities using w1=0.5, w2=0.5[cite: 105, 111]."""
    combined = []
    # Formula: combined = w1 * video_prob + w2 * audio_prob [cite: 110]
    for v, a in zip(video_probs["video"], audio_probs["audio"]):
        v_p = v["probability"]
        a_p = a["probability"]
        c_p = (0.5 * v_p) + (0.5 * a_p)
        
        combined.append({
            "start": v["start"],
            "end": v["end"],
            "video_probability": v_p,
            "audio_probability": a_p,
            "combined_probability": round(c_p, 4)
        })
    return combined

if __name__ == "__main__":
    INPUT_FILE = "input.mp4" 
    if os.path.exists(INPUT_FILE):
        print(f"Starting session for {INPUT_FILE}...")
        results = run_fake_detection_session(INPUT_FILE)
        print("\n--- Final Response (Frontend Ready) ---")
        print(json.dumps(results, indent=2))
    else:
        print(f"Error: '{INPUT_FILE}' not found in the current directory.")