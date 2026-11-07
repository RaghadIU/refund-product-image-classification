
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import statistics


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
LOG_PATH = RESULTS_DIR / "monitoring_log.jsonl"


def main():
    csv_files = sorted(
        RESULTS_DIR.glob("batch_predictions_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        print("No batch prediction CSV files found.")
        return

    latest_file = csv_files[0]

    with latest_file.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    successful = [row for row in rows if row.get("status") == "success"]
    errors = [row for row in rows if row.get("status") == "error"]

    confidences = []
    for row in successful:
        try:
            confidences.append(float(row["confidence"]))
        except (ValueError, TypeError, KeyError):
            pass

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": latest_file.name,
        "images_in_results": len(rows),
        "successful_predictions": len(successful),
        "errors": len(errors),
        "success_rate": (
            round(len(successful) / len(rows), 4) if rows else 0.0
        ),
        "average_confidence": (
            round(statistics.mean(confidences), 4)
            if confidences else None
        ),
        "low_confidence_predictions_below_0_60": sum(
            confidence < 0.60 for confidence in confidences
        ),
    }

    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nMonitoring log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
