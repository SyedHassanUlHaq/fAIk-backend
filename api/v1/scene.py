from ml_models.scene_detection import get_embedding_model
from services.scene_detection.detector import detect_scene_changes
from services.scene_detection.video_utils import convert_to_fps
from fastapi import APIRouter, Depends

router = APIRouter()

@router.post("/detect-scenes")
def detect_scenes(video_path: str, pipe=Depends(get_embedding_model)):
    converted = convert_to_fps(video_path)
    results = detect_scene_changes(converted, pipe)

    return {
        "video": video_path,
        "cuts": results,
        "count": len(results)
    }