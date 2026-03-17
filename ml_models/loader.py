# ml_models/loader.py

from transformers import XCLIPVisionModel
from validation.models.demamba.DeMamba import XCLIP_DeMamba
from validation.models.fused_model import FusedHeadModel
from validation.validate import load_raft_model
from config.project_config import CHECKPOINT, DEVICE

# Global model references
raft_model = None
fused_model = None
xclip_demamba = None
clip_model = None
clip_preprocess = None

def load_models():
    global raft_model, fused_model, xclip_demamba, clip_model, clip_preprocess

    # --------------------------
    # RAFT
    # --------------------------
    if raft_model is None:
        print("[*] Loading RAFT model...")
        raft_model = load_raft_model("validation/checkpoints/raft-sintel.pth", DEVICE)
        print("[+] RAFT model loaded")

    # --------------------------
    # XCLIP Vision
    # --------------------------
    if xclip_demamba is None:
        print("[*] Loading XCLIPVisionModel + DeMamba...")
        xclip_encoder = XCLIPVisionModel.from_pretrained(
            "microsoft/xclip-base-patch16"
        ).to(DEVICE)
        xclip_encoder.eval()
        xclip_demamba = XCLIP_DeMamba(pretrained_encoder=xclip_encoder).to(DEVICE)
        xclip_demamba.eval()
        print("[+] XCLIPVisionModel + DeMamba loaded")

    # --------------------------
    # FusedHeadModel
    # --------------------------
    if fused_model is None:
        print("[*] Loading FusedHeadModel...")
        fused_model = FusedHeadModel(pretrained_xclip_encoder=xclip_demamba).to(DEVICE)
        fused_model.load_checkpoint(CHECKPOINT)
        fused_model.eval()
        print("[+] FusedHeadModel loaded")


    return raft_model, fused_model, xclip_demamba