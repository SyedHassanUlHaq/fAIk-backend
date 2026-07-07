"""
Smoke test for ai_models_service (voice / audio AI only — scene/tamper
detection now lives in its own top-level project at ../scene_detection).

Run from inside ai_models_service/ (so local packages resolve the same way
they do for `uvicorn main:app`):

    cd ai_models_service
    python test.py

Checks are split into two tiers:
  - HARD checks: pure code/wiring correctness (imports, Celery task
    registration, route wiring). No network/DB required — these must pass.
  - SOFT checks: anything that touches the outside world (a live
    DATABASE_URL). Reported but don't fail the run.
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
# HARD checks — no network, no DB
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


def check_helpers_import():
    from helpers.audio_helper import extract_audio_from_video, split_audio_into_intervals  # noqa: F401


def check_api_router_import():
    from api import voice
    assert hasattr(voice, "router"), "api.voice has no `router`"


def check_fastapi_app_builds():
    import main
    assert main.app is not None
    route_paths = {r.path for r in main.app.routes}
    assert "/" in route_paths, "root route missing"
    assert "/v1/voice/health" in route_paths, "voice router not mounted"


def check_celery_task_registration():
    from celery_app import celery_app
    import tasks.audio_scan_task  # noqa: F401
    assert "tasks.process_audio_scan" in celery_app.tasks


def check_voice_health_endpoint():
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    resp = client.get("/v1/voice/health")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "active"}


def check_root_endpoint():
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    resp = client.get("/")
    assert resp.status_code == 200, resp.text
    assert resp.json()["message"] == "fAIk Voice AI Service running"


# ---------------------------------------------------------------------------
# SOFT checks — touch the network / a real database
# ---------------------------------------------------------------------------

def check_database_connection():
    from sqlalchemy import text
    from database import engine
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main_test() -> int:
    print("=== ai_models_service (voice/audio) smoke test ===\n")

    print("-- hard checks (imports & wiring, no network/DB required) --")
    _run("config.project_config imports", check_config_imports)
    _run("database module imports", check_database_module_imports)
    _run("models (User, Scan) import", check_models_import)
    _run("utils (errors, s3, push) import", check_utils_import)
    _run("helpers.audio_helper imports", check_helpers_import)
    _run("api.voice router imports", check_api_router_import)
    _run("FastAPI app builds + routes mounted", check_fastapi_app_builds)
    _run("Celery task registers (audio)", check_celery_task_registration)
    _run("GET /v1/voice/health", check_voice_health_endpoint)
    _run("GET /", check_root_endpoint)

    print("\n-- soft checks (network / database required) --")
    _run("database connection (SELECT 1)", check_database_connection, hard=False)

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
