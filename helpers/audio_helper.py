import os
from moviepy import AudioFileClip, VideoFileClip

def extract_audio_from_video(video_path: str, folder_name: str) -> str:
    """Extracts full audio track from video."""
    output_path = f"{folder_name}/audios/original_audio.wav"
    with VideoFileClip(video_path) as video:
        if video.audio is not None:
            video.audio.write_audiofile(output_path, logger=None)
    return output_path

def split_audio_into_intervals(audio_path: str, folder_name: str) -> list[str]:
    """Splits an audio file into 5-second chunks."""
    paths = []
    if not os.path.exists(audio_path):
        return paths

    with AudioFileClip(audio_path) as audio:
        duration = int(audio.duration)
        for start in range(0, duration, 5):
            end = min(start + 5, duration)
            target = f"{folder_name}/audios/audio_{start:03d}.wav"
            subclip = audio.subclip(start, end)
            subclip.write_audiofile(target, logger=None)
            paths.append(target)
    return paths
