import os
from moviepy.editor import VideoFileClip

def split_video_into_intervals(video_path: str, folder_name: str) -> list[str]:
    """Splits a video into 5-second chunks."""
    paths = []
    with VideoFileClip(video_path) as clip:
        duration = int(clip.duration)
        for start in range(0, duration, 5):
            end = min(start + 5, duration)
            target = f"{folder_name}/videos/video_{start:03d}.mp4"
            subclip = clip.subclip(start, end)
            subclip.write_videofile(target, codec="libx264", audio_codec="aac", logger=None)
            paths.append(target)
    return paths
