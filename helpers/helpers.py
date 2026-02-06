# import os
# import uuid
# import json
# from typing import List, Dict
# from moviepy import VideoFileClip, AudioFileClip 
# from validation.validate import load_raft_model

# audio_model = load_raft_model(ckpt_path="checkpoints\fused_best.pt", device="cpu")


# def run_fake_detection_session(video_path: str) -> Dict:
#     """End-to-end orchestration for frontend and API usage[cite: 124]."""
    

#     session_id = str(uuid.uuid4())
#     folder_name = f"sessions/{session_id}"
    
#     os.makedirs(f"{folder_name}/original", exist_ok=True)
#     os.makedirs(f"{folder_name}/videos", exist_ok=True)
#     os.makedirs(f"{folder_name}/audios", exist_ok=True)
#     os.makedirs(f"{folder_name}/results", exist_ok=True)

#     original_audio_path = extract_audio_from_video(video_path, folder_name)
    
#     split_video_into_intervals(video_path, folder_name)
#     split_audio_into_intervals(original_audio_path, folder_name)
    
#     video_probs = predict_all_video_intervals(folder_name)
#     audio_probs = predict_all_audio_intervals(folder_name)
    
#     combined_results = combine_interval_probabilities(video_probs, audio_probs)
    
#     final_response = {
#         "session_id": session_id,
#         "intervals": combined_results
#     }

#     with open(f"{folder_name}/results/probabilities.json", "w") as f:
#         json.dump(final_response, f, indent=4)
        
#     return final_response

# def extract_audio_from_video(video_path: str, folder_name: str) -> str:
#     """Extracts full audio track from video[cite: 98]."""
#     output_path = f"{folder_name}/audios/original_audio.wav"
#     with VideoFileClip(video_path) as video:
#         if video.audio is not None:
#             video.audio.write_audiofile(output_path, logger=None)
#     return output_path

# def split_video_into_intervals(video_path: str, folder_name: str) -> List[str]:
#     """Splits a video into 5-second chunks[cite: 25]."""
#     paths = []
#     with VideoFileClip(video_path) as clip:
#         duration = int(clip.duration)
#         for start in range(0, duration, 5):
#             end = min(start + 5, duration)
#             target = f"{folder_name}/videos/video_{start:03d}.mp4"
#             subclip = clip.subclipped(start, end)
#             subclip.write_videofile(target, codec="libx264", audio_codec="aac", logger=None)
#             paths.append(target)
#     return paths

# def split_audio_into_intervals(audio_path: str, folder_name: str) -> List[str]:
#     """Splits an audio file into 5-second chunks[cite: 38]."""
#     paths = []
#     if not os.path.exists(audio_path): return paths
    
#     with AudioFileClip(audio_path) as audio:
#         duration = int(audio.duration)
#         for start in range(0, duration, 5):
#             end = min(start + 5, duration)
#             target = f"{folder_name}/audios/audio_{start:03d}.wav"
#             subclip = audio.subclipped(start, end)
#             subclip.write_audiofile(target, logger=None)
#             paths.append(target)
#     return paths

# def predict_all_video_intervals(folder_path: str) -> Dict:
#     """Runs the video model on all video intervals[cite: 52]."""
#     results = []
#     video_dir = f"{folder_path}/videos/"
#     files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
#     for i, filename in enumerate(files):

#         prob = predict_video(os.path.join(video_dir, filename))
#         results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
#     return {"video": results}

# def predict_all_audio_intervals(folder_path: str) -> Dict:
#     """Runs the audio model on all audio intervals[cite: 71]."""
#     results = []
#     audio_dir = f"{folder_path}/audios/"

#     files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav") and "original" not in f])
#     for i, filename in enumerate(files):

#         prob = predict_audio(os.path.join(audio_dir, filename))
#         results.append({"start": i*5, "end": (i+1)*5, "probability": prob})
#     return {"audio": results}


# def combine_interval_probabilities(video_probs: Dict, audio_probs: Dict) -> List[Dict]:
#     """Combines audio and video probabilities using w1=0.5, w2=0.5[cite: 105, 111]."""
#     combined = []

#     for v, a in zip(video_probs["video"], audio_probs["audio"]):
#         v_p = v["probability"]
#         a_p = a["probability"]
#         c_p = (0.5 * v_p) + (0.5 * a_p)
        
#         combined.append({
#             "start": v["start"],
#             "end": v["end"],
#             "video_probability": v_p,
#             "audio_probability": a_p,
#             "combined_probability": round(c_p, 4)
#         })
#     return combined

# if __name__ == "__main__":
#     INPUT_FILE = "input.mp4" 
#     if os.path.exists(INPUT_FILE):
#         print(f"Starting session for {INPUT_FILE}...")
#         results = run_fake_detection_session(INPUT_FILE)
#         print("\n--- Final Response (Frontend Ready) ---")
#         print(json.dumps(results, indent=2))
#     else:
#         print(f"Error: '{INPUT_FILE}' not found in the current directory.")
