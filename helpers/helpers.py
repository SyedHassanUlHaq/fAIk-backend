# This file is deprecated. Use individual helper modules instead:
# - audio_helper.py for audio operations
# - video_helper.py for video operations  
# - prediction_helper.py for ML predictions



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
