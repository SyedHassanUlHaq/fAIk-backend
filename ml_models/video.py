import threading
import logging
from transformers import XCLIPVisionModel
from repositories.validation_tool.models.demamba.DeMamba import XCLIP_DeMamba
from repositories.validation_tool.models.fused_model import FusedHeadModel
from repositories.validation_tool.validate import load_raft_model
from config.project_config import CHECKPOINT, DEVICE

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()

def load_models():
    global raft_model, fused_model, xclip_demamba

    with _load_lock:
        try:
            if raft_model is None:
                print("[*] Loading RAFT model...")
                raft_model = load_raft_model("repositories/validation_tool/checkpoints/raft-sintel.pth", DEVICE)
                print("[+] RAFT model loaded")

            if xclip_demamba is None:
                print("[*] Loading XCLIPVisionModel + DeMamba...")
                xclip_encoder = XCLIPVisionModel.from_pretrained(
                    "microsoft/xclip-base-patch16"
                ).to(DEVICE)
                xclip_encoder.eval()
                xclip_demamba = XCLIP_DeMamba(pretrained_encoder=xclip_encoder).to(DEVICE)
                xclip_demamba.eval()
                print("[+] XCLIPVisionModel + DeMamba loaded")

            if fused_model is None:
                print("[*] Loading FusedHeadModel...")
                fused_model = FusedHeadModel(pretrained_xclip_encoder=xclip_demamba).to(DEVICE)
                fused_model.load_checkpoint(CHECKPOINT)
                fused_model.eval()
                print("[+] FusedHeadModel loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise

    return raft_model, fused_model, xclip_demamba