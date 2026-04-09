import os
import logging
from moviepy import AudioFileClip, VideoFileClip

logger = logging.getLogger(__name__)

def extract_audio_from_video(video_path: str, folder_name: str) -> str:
    """Extracts full audio track from video."""
    try:
        output_path = f"{folder_name}/audios/original_audio.wav"
        with VideoFileClip(video_path) as video:
            if video.audio is not None:
                video.audio.write_audiofile(output_path, logger=None)
        return output_path
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}", exc_info=True)
        raise

def split_audio_into_intervals(audio_path: str, folder_name: str) -> list[str]:
    """Splits an audio file into 5-second chunks."""
    paths = []
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return paths

        with AudioFileClip(audio_path) as audio:
            duration = int(audio.duration)
            for start in range(0, duration, 5):
                end = min(start + 5, duration)
                target = f"{folder_name}/audios/audio_{start:03d}.wav"
                subclip = audio.subclip(start, end)
                subclip.write_audiofile(target, logger=None)
                paths.append(target)
    except Exception as e:
        logger.error(f"Failed to split audio {audio_path}: {e}", exc_info=True)
        raise
    
    return paths
