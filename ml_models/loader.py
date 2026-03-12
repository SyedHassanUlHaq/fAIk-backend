# models.py
from validation.validate import load_raft_model
from validation.models.fused_model import FusedHeadModel
from config.project_config import CHECKPOINT, DEVICE

raft_model = None
fused_model = None

def load_models():
    global raft_model, fused_model

    if raft_model is None:
        raft_model = load_raft_model("validation/checkpoints/raft-sintel.pth", DEVICE)
        print("[+] RAFT model loaded")

    if fused_model is None:
        fused_model = FusedHeadModel().to(DEVICE)
        fused_model.load_checkpoint(CHECKPOINT)
        fused_model.eval()
        print("[+] FusedHeadModel loaded")