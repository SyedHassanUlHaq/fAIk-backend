# Handles audio/video predictions and combining results.
import os
from typing import Dict
from validation.validate import load_raft_model

audio_model = load_raft_model(ckpt_path="checkpoints/fused_best.pt", device="cpu")

def predict_video(video_path: str) -> float:
    """Predicts deepfake probability for a video interval."""
    # TODO: Implement video prediction logic
    return 0.0

def predict_audio(audio_path: str) -> float:
    """Predicts deepfake probability for an audio interval."""
    # TODO: Implement audio prediction logic
    return 0.0

def predict_all_video_intervals(folder_path: str) -> Dict:
    results = []
    video_dir = f"{folder_path}/videos/"
    files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    for i, filename in enumerate(files):
        prob = predict_video(os.path.join(video_dir, filename))
        results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
    return {"video": results}

def predict_all_audio_intervals(folder_path: str) -> Dict:
    results = []
    audio_dir = f"{folder_path}/audios/"
    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav") and "original" not in f])
    for i, filename in enumerate(files):
        prob = predict_audio(os.path.join(audio_dir, filename))
        results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
    return {"audio": results}

def combine_interval_probabilities(video_probs: Dict, audio_probs: Dict) -> list[Dict]:
    """Combines audio and video probabilities using w1=0.5, w2=0.5."""
    combined = []
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
