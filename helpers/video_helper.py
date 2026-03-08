import os
from moviepy.editor import VideoFileClip

def split_video_into_intervals(video_path: str, folder_name: str) -> list[str]:
    """Split video into 5-second chunks."""

    videos_dir = os.path.join(folder_name, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    paths = []

    with VideoFileClip(video_path) as clip:
        duration = int(clip.duration)

        for start in range(0, duration, 5):
            end = min(start + 5, duration)

            target = os.path.join(videos_dir, f"video_{start:03d}.mp4")

            subclip = clip.subclip(start, end)
            subclip.write_videofile(
                target,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )

            subclip.close()

            paths.append(target)

    return paths