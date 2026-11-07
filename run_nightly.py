
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

APP_FILE = BASE_DIR / "app.py"
PREPARE_FILE = BASE_DIR / "prepare_daily_images.py"
BATCH_FILE = BASE_DIR / "batch_predict.py"
MONITOR_FILE = BASE_DIR / "monitoring.py"

LOG_DIR = BASE_DIR / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "nightly_run.log"

HEALTH_URL = "http://127.0.0.1:5000/health"


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def api_is_running():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def start_api_if_needed():
    if api_is_running():
        log("Flask API is already running.")
        return

    log("Flask API is not running. Starting it...")

    with open(LOG_FILE, "a", encoding="utf-8") as app_log:
        subprocess.Popen(
            [sys.executable, str(APP_FILE)],
            cwd=str(BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=app_log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    for attempt in range(30):
        time.sleep(2)
        if api_is_running():
            log("Flask API is ready.")
            return

    raise RuntimeError(
        "Flask API did not become ready within 60 seconds."
    )


def run_script(script_path):
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    log(f"Starting {script_path.name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.stdout:
        log(result.stdout.strip())

    if result.stderr:
        log(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code "
            f"{result.returncode}."
        )

    log(f"{script_path.name} completed successfully.")


def main():
    log("========== NIGHTLY BATCH RUN STARTED ==========")

    try:
        # Step 1: Select and copy 500 previously unused images.
        run_script(PREPARE_FILE)

        # Step 2: Start the API if it is not already running.
        start_api_if_needed()

        # Step 3: Classify the new images.
        run_script(BATCH_FILE)

        # Step 4: Record monitoring metrics.
        run_script(MONITOR_FILE)

        log("NIGHTLY BATCH RUN COMPLETED SUCCESSFULLY.")

    except Exception as error:
        log(f"ERROR: {error}")
        sys.exit(1)

    finally:
        log("========== NIGHTLY BATCH RUN FINISHED ==========")


if __name__ == "__main__":
    main()
