"""
Smoke test for scene_detection.

Run from inside scene_detection/ (so local packages resolve the same way
they do for `uvicorn main:app`):

    cd scene_detection
    python test.py

Checks are split into two tiers:
  - HARD checks: pure code/wiring correctness (imports, Celery task
    registration, route wiring). No network/GPU required — these must pass.
  - SOFT checks: anything that touches the outside world (downloading the
    embedding model from Hugging Face, a live DATABASE_URL). These are
    reported but don't fail the run, since a missing network connection or
    unset DB isn't a bug in this service's code.
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
# HARD checks — no network, no GPU
# ---------------------------------------------------------------------------

def check_config_imports():
    from config.project_config import DATABASE_URL, REDIS_URL, AWS_BUCKET_NAME  # noqa: F401


def check_database_module_imports():
    from database import Base, SessionLocal, engine, get_db  # noqa: F401


def check_models_import():
    from models import User, Scan  # noqa: F401


def check_utils_import():
    from utils.errors import AppError  # noqa: F401
    from utils.s3 import upload_file, download_file, delete_file, presigned_url  # noqa: F401
    from utils.push import send_push  # noqa: F401


def check_ml_models_module_imports():
    # Importing must succeed without triggering a model download —
    # get_embedding_model() only downloads/loads on first *call*.
    from ml_models.scene_detection import get_embedding_model  # noqa: F401
    from ml_models import get_embedding_model as get_embedding_model_reexport  # noqa: F401


def check_scene_detection_services_import():
    from services.scene_detection.detector import detect_scene_changes  # noqa: F401
    from services.scene_detection.embeddings import compute_embedding, compute_embeddings_batch, cosine_similarity  # noqa: F401
    from services.scene_detection.video_utils import convert_to_fps  # noqa: F401


def check_api_router_import():
    from api import scene
    assert hasattr(scene, "router"), "api.scene has no `router`"


def check_fastapi_app_builds():
    """Build the FastAPI app object without running lifespan (no model download)."""
    import main
    assert main.app is not None
    route_paths = {r.path for r in main.app.routes}
    assert "/" in route_paths, "root route missing"
    assert "/v1/scene/detect-scenes" in route_paths, "scene router not mounted"


def check_celery_task_registration():
    from celery_app import celery_app
    import tasks.tamper_scan_task  # noqa: F401
    assert "tasks.process_tamper_scan" in celery_app.tasks


# ---------------------------------------------------------------------------
# SOFT checks — touch the network / a real database
# ---------------------------------------------------------------------------

def check_database_connection():
    from sqlalchemy import text
    from database import engine
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def check_embedding_model_loads():
    """Actually downloads/loads nomic-embed-vision-v1.5 — requires network (and ideally a GPU)."""
    from ml_models.scene_detection import get_embedding_model
    model, processor, device = get_embedding_model()
    assert model is not None and processor is not None
    print(f"      embedding model loaded on device={device}")


def check_root_and_lifespan_endpoint():
    """Full startup including the embedding-model lifespan — requires network."""
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "fAIk Scene Detection Service running"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main_test() -> int:
    print("=== scene_detection smoke test ===\n")

    print("-- hard checks (imports & wiring, no network required) --")
    _run("config.project_config imports", check_config_imports)
    _run("database module imports", check_database_module_imports)
    _run("models (User, Scan) import", check_models_import)
    _run("utils (errors, s3, push) import", check_utils_import)
    _run("ml_models.scene_detection imports", check_ml_models_module_imports)
    _run("services.scene_detection imports", check_scene_detection_services_import)
    _run("api.scene router imports", check_api_router_import)
    _run("FastAPI app builds + routes mounted", check_fastapi_app_builds)
    _run("Celery task registers (tamper)", check_celery_task_registration)

    print("\n-- soft checks (network / database required) --")
    _run("database connection (SELECT 1)", check_database_connection, hard=False)
    _run("embedding model loads (nomic-embed-vision-v1.5)", check_embedding_model_loads, hard=False)
    _run("GET / with full lifespan startup", check_root_and_lifespan_endpoint, hard=False)

    print("\n=== summary ===")
    if hard_failures:
        print(f"{FAIL} {len(hard_failures)} hard check(s) failed:")
        for f in hard_failures:
            print(f"  - {f}")
    else:
        print(f"{PASS} all hard checks passed")

    if soft_warnings:
        print(f"{WARN} {len(soft_warnings)} soft check(s) skipped/failed (network or DB not available):")
        for w in soft_warnings:
            print(f"  - {w}")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main_test())
