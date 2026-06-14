#!/usr/bin/env python3
"""
Profile RAM and GPU VRAM consumed by each model in the fAIk pipeline.

Models measured:
  1. RAFT (large)              - optical flow
  2. XCLIPVisionModel          - microsoft/xclip-base-patch16
  3. XCLIP_DeMamba             - XCLIPVisionModel + Mamba blocks
  4. OpticalFlowBranch         - ResNet-50 backbone (2-channel input)
  5. FusedHeadModel            - XCLIP_DeMamba + OpticalFlowBranch + fusion head
  6. nomic-embed-vision-v1.5   - scene-detection embedding model

Usage:
    python profile_models.py [--device cuda|cpu] [--raft-ckpt PATH]
"""

import argparse
import gc
import os
import sys
import time

import psutil
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def _vram_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _params_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def _measure(tag: str, device: str, loader_fn):
    """
    Call loader_fn(), measure RAM and VRAM delta, return (model, result_dict).
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_before  = _ram_mb()
    vram_before = _vram_mb(device)

    t0 = time.perf_counter()
    model = loader_fn()
    elapsed = time.perf_counter() - t0

    if device == "cuda":
        torch.cuda.synchronize()

    ram_after  = _ram_mb()
    vram_after = _vram_mb(device)

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024 ** 2 if device == "cuda" else 0.0
    )

    params_mb = _params_mb(model) if model is not None else 0.0

    return model, {
        "tag":        tag,
        "params_mb":  params_mb,
        "ram_delta":  ram_after - ram_before,
        "vram_delta": vram_after - vram_before,
        "vram_peak":  peak_vram,
        "load_s":     elapsed,
    }


def _print_row(r: dict):
    print(
        f"  {'RAM Δ':>10} {r['ram_delta']:>8.1f} MB"
        f"  |  {'GPU Δ':>8} {r['vram_delta']:>8.1f} MB"
        f"  (peak {r['vram_peak']:>8.1f} MB)"
        f"  |  params {r['params_mb']:>8.1f} MB"
        f"  |  load {r['load_s']:.1f}s"
    )


def _separator():
    print("─" * 90)


# ── loaders ──────────────────────────────────────────────────────────────────

def load_raft(ckpt_path: str, device: str):
    from repositories.validation_tool.validate import load_raft_model
    if not os.path.exists(ckpt_path):
        print(f"    [!] checkpoint not found: {ckpt_path} — skipping load, counting params only")
        from repositories.validation_tool.raft.raft import RAFT

        class AttrDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        args = AttrDict({"small": False, "mixed_precision": False, "dropout": 0.0,
                          "alternate_corr": False, "corr_levels": 4, "corr_radius": 4})
        model = RAFT(args).to(device)
        model.eval()
        return model
    return load_raft_model(ckpt_path, device)


def load_xclip(device: str):
    from transformers import XCLIPVisionModel
    model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16").to(device)
    model.eval()
    return model


def load_xclip_demamba(xclip_encoder, device: str):
    from repositories.validation_tool.models.demamba.DeMamba import XCLIP_DeMamba
    model = XCLIP_DeMamba(pretrained_encoder=xclip_encoder).to(device)
    model.eval()
    return model


def load_optical_flow_branch(device: str):
    from repositories.validation_tool.models.optical_flow_model import OpticalFlowBranch
    model = OpticalFlowBranch(pretrained=False, backbone="resnet50").to(device)
    model.eval()
    return model


def load_fused(xclip_demamba, device: str):
    from repositories.validation_tool.models.fused_model import FusedHeadModel
    model = FusedHeadModel(pretrained_xclip_encoder=xclip_demamba).to(device)
    model.eval()
    return model


def load_nomic(device: str):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to(device)
    model.eval()
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile model RAM / GPU usage")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--raft-ckpt", default="repositories/validation_tool/checkpoints/raft-sintel.pth")
    args = parser.parse_args()

    device = args.device

    print(f"\n{'═'*90}")
    print(f"  fAIk Model Memory Profiler")
    print(f"  Device : {device.upper()}")
    if device == "cuda":
        prop = torch.cuda.get_device_properties(0)
        total_vram = prop.total_memory / 1024 ** 2
        print(f"  GPU    : {prop.name}  ({total_vram:.0f} MB VRAM)")
    ram_total = psutil.virtual_memory().total / 1024 ** 2
    print(f"  RAM    : {ram_total:.0f} MB total  |  available {psutil.virtual_memory().available/1024**2:.0f} MB")
    print(f"{'═'*90}\n")

    results = []

    # 1 — RAFT
    _separator()
    print(f"[1/6]  RAFT (large)  —  optical flow model")
    raft_model, r = _measure("RAFT", device, lambda: load_raft(args.raft_ckpt, device))
    _print_row(r);  results.append(r)

    # 2 — XCLIPVisionModel (standalone, before DeMamba wrapping)
    _separator()
    print(f"[2/6]  XCLIPVisionModel  —  microsoft/xclip-base-patch16")
    xclip_model, r = _measure("XCLIPVisionModel", device, lambda: load_xclip(device))
    _print_row(r);  results.append(r)

    # 3 — XCLIP_DeMamba (reuses already-loaded xclip_model)
    _separator()
    print(f"[3/6]  XCLIP_DeMamba  —  XCLIPVisionModel + Mamba blocks")
    print(f"       (wraps the encoder above; delta shows only the NEW Mamba layers)")
    demamba_model, r = _measure("XCLIP_DeMamba", device, lambda: load_xclip_demamba(xclip_model, device))
    _print_row(r);  results.append(r)

    # 4 — OpticalFlowBranch (ResNet-50)
    _separator()
    print(f"[4/6]  OpticalFlowBranch  —  ResNet-50 (2-channel input, no pretrained weights)")
    flow_model, r = _measure("OpticalFlowBranch", device, lambda: load_optical_flow_branch(device))
    _print_row(r);  results.append(r)

    # 5 — FusedHeadModel (reuses demamba_model + adds flow branch + tiny head)
    _separator()
    print(f"[5/6]  FusedHeadModel  —  XCLIP_DeMamba + OpticalFlowBranch + fusion head")
    print(f"       (wraps models above; delta = fusion head only)")
    fused_model, r = _measure("FusedHeadModel", device, lambda: load_fused(demamba_model, device))
    _print_row(r);  results.append(r)

    # 6 — nomic-embed-vision-v1.5
    _separator()
    print(f"[6/6]  nomic-embed-vision-v1.5  —  scene-detection embedding model")
    nomic_model, r = _measure("nomic-embed-vision-v1.5", device, lambda: load_nomic(device))
    _print_row(r);  results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    _separator()
    print(f"\n{'MODEL':<28} {'PARAMS':>10} {'RAM Δ':>10} {'GPU Δ':>10} {'GPU PEAK':>10}  LOAD")
    print(f"{'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*6}")
    for r in results:
        print(
            f"  {r['tag']:<26} {r['params_mb']:>8.1f}MB"
            f"  {r['ram_delta']:>8.1f}MB"
            f"  {r['vram_delta']:>8.1f}MB"
            f"  {r['vram_peak']:>8.1f}MB"
            f"  {r['load_s']:.1f}s"
        )

    total_vram_delta = sum(r["vram_delta"] for r in results)
    total_ram_delta  = sum(r["ram_delta"]  for r in results)
    print(f"\n  {'TOTAL (cumulative Δ)':<26} {'':>8}   {total_ram_delta:>8.1f}MB  {total_vram_delta:>8.1f}MB")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 2
        print(f"\n  GPU memory now:  allocated {allocated:.1f} MB  |  reserved {reserved:.1f} MB")

    print(f"\n{'═'*90}\n")


if __name__ == "__main__":
    main()
