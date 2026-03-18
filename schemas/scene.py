from pydantic import BaseModel
from typing import List

class SceneCut(BaseModel):
    frame: int
    ssim: float
    mse: float
    emb_diff: float

class SceneResponse(BaseModel):
    video: str
    cuts: List[SceneCut]
    count: int