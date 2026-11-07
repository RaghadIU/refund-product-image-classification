
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import requests


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data" / "incoming"
RESULTS_DIR = PROJECT_DIR / "results"

API_URL = "http://127.0.0.1:5000/predict/batch"
BATCH_SIZE = 32
TIMEOUT_SECONDS = 180

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"batch_predictions_{timestamp}.csv"
    error_count = 0
    processed_count = 0

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "filename",
            "predicted_category",
            "confidence",
            "probabilities",
            "status",
            "error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[start:start + BATCH_SIZE]
            opened_files = []

            try:
                for path in batch_paths:
                    file_handle = path.open("rb")
                    opened_files.append(file_handle)

                files = [
                    ("images", (path.name, handle, "application/octet-stream"))
                    for path, handle in zip(batch_paths, opened_files)
                ]

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()

                for result in payload.get("results", []):
                    probabilities = result.get("probabilities", {})
                    confidence = max(probabilities.values(), default=0.0)
                    status = "success" if "error" not in result else "error"

                    if status == "error":
                        error_count += 1

                    writer.writerow({
                        "filename": result.get("filename", ""),
                        "predicted_category": result.get(
                            "predicted_category", ""
                        ),
                        "confidence": confidence,
                        "probabilities": json.dumps(
                            probabilities, ensure_ascii=False
                        ),
                        "status": status,
                        "error": result.get("error", ""),
                    })
                    processed_count += 1

            except (requests.RequestException, ValueError) as error:
                error_count += len(batch_paths)
                for path in batch_paths:
                    writer.writerow({
                        "filename": path.name,
                        "predicted_category": "",
                        "confidence": "",
                        "probabilities": "",
                        "status": "error",
                        "error": str(error),
                    })

            finally:
                for handle in opened_files:
                    handle.close()

    print(f"Images found: {len(image_paths)}")
    print(f"Results written: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
