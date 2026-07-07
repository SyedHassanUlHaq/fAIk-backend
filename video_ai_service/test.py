"""
Smoke test for video_ai_service.

Run from inside video_ai_service/ (so local packages resolve the same way
they do for `uvicorn main:app`):

    cd video_ai_service
    python test.py

Checks are split into two tiers:
  - HARD checks: pure code/wiring correctness (imports, Celery task
    registration, route wiring). No network/GPU/model-weights required —
    these must pass.
  - SOFT checks: anything that needs the actual model weights on disk
    (checkpoints/fused_best.pt, repositories/validation_tool/checkpoints/raft-sintel.pth,
    microsoft/xclip-base-patch16/) or a live DATABASE_URL. Reported but
    don't fail the run, since missing weights/DB isn't a code bug.
"""

import sys
import traceback


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

hard_failures: list[str] = []
soft_warnings: list[str] = []


def _run(label: str, fn, hard: bool = True):
    try:
        fn()
        print(f"{PASS} {label}")
        return True
    except Exception as exc:
        bucket = hard_failures if hard else soft_warnings
        bucket.append(f"{label}: {exc}")
        print(f"{FAIL if hard else WARN} {label}: {exc}")
        if hard:
            traceback.print_exc(limit=3)
        return False


# ---------------------------------------------------------------------------
# HARD checks — no network, no GPU, no model weights required
# ---------------------------------------------------------------------------

def check_config_imports():
    from config.project_config import (  # noqa: F401
        DATABASE_URL, REDIS_URL, UPLOAD_DIR, CHECKPOINT, DEVICE, THRESHOLD, MODEL_VERSION,
    )


def check_database_module_imports():
    from database import Base, SessionLocal, engine, get_db  # noqa: F401


def check_models_import():
    from models import User, Scan  # noqa: F401


def check_utils_import():
    from utils.errors import AppError  # noqa: F401
    from utils.s3 import upload_file, download_file, delete_file, presigned_url  # noqa: F401
    from utils.push import send_push  # noqa: F401


def check_helpers_import():
    from helpers.video_helper import split_video_into_chunks, infer_chunk  # noqa: F401


def check_validation_tool_imports():
    """
    Importing must succeed without touching disk/network — it only defines
    classes (RAFT, FusedHeadModel, ValidationTransform) and wires up
    fully-qualified `repositories.validation_tool.*` imports. Actual weight
    loading only happens inside load_models().
    """
    from repositories.validation_tool import validate_video  # noqa: F401
    from ml_models.video import load_models  # noqa: F401


def check_api_router_import():
    from api import video
    assert hasattr(video, "router"), "api.video has no `router`"


def check_fastapi_app_builds():
    """Importing main.py builds the app/routes but does NOT run lifespan (no model load)."""
    import main
    assert main.app is not None
    route_paths = {r.path for r in main.app.routes}
    assert "/" in route_paths, "root route missing"
    assert "/v1/video/upload-video" in route_paths, "video router not mounted"
    assert "/v1/video/results/{session_id}" in route_paths
    assert "/v1/video/health" in route_paths


def check_celery_task_registration():
    from celery_app import celery_app
    import tasks.video_scan_task  # noqa: F401
    assert "tasks.process_video_scan" in celery_app.tasks


# ---------------------------------------------------------------------------
# SOFT checks — need real model weights on disk / a live database
# ---------------------------------------------------------------------------

def check_database_connection():
    from sqlalchemy import text
    from database import engine
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def check_models_load():
    """
    Loads RAFT + XCLIP-DeMamba + FusedHeadModel for real. Needs:
      - checkpoints/fused_best.pt
      - repositories/validation_tool/checkpoints/raft-sintel.pth
      - microsoft/xclip-base-patch16/ (local HF snapshot, or network access)
    """
    from ml_models.video import load_models
    raft_model, fused_model, xclip_demamba = load_models()
    assert raft_model is not None
    assert fused_model is not None
    assert xclip_demamba is not None


def check_root_and_lifespan_endpoint():
    """Full startup including load_models() in the lifespan — needs the weights above."""
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "fAIk Video AI Service running"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main_test() -> int:
    print("=== video_ai_service smoke test ===\n")

    print("-- hard checks (imports & wiring, no weights/network required) --")
    _run("config.project_config imports", check_config_imports)
    _run("database module imports", check_database_module_imports)
    _run("models (User, Scan) import", check_models_import)
    _run("utils (errors, s3, push) import", check_utils_import)
    _run("helpers.video_helper imports", check_helpers_import)
    _run("repositories.validation_tool / ml_models.video import", check_validation_tool_imports)
    _run("api.video router imports", check_api_router_import)
    _run("FastAPI app builds + routes mounted", check_fastapi_app_builds)
    _run("Celery task registers (video)", check_celery_task_registration)

    print("\n-- soft checks (model weights on disk / database required) --")
    _run("database connection (SELECT 1)", check_database_connection, hard=False)
    _run("load_models() — RAFT + XCLIP-DeMamba + FusedHeadModel", check_models_load, hard=False)
    _run("GET / with full lifespan startup (loads all models)", check_root_and_lifespan_endpoint, hard=False)

    print("\n=== summary ===")
    if hard_failures:
        print(f"{FAIL} {len(hard_failures)} hard check(s) failed:")
        for f in hard_failures:
            print(f"  - {f}")
    else:
        print(f"{PASS} all hard checks passed")

    if soft_warnings:
        print(f"{WARN} {len(soft_warnings)} soft check(s) skipped/failed (weights/network/DB not available):")
        for w in soft_warnings:
            print(f"  - {w}")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main_test())
