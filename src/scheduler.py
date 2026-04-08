"""Automated monitoring scheduler with alerting for production drift detection.

Modes
-----
once  : Process all unprocessed batches in data/production/, then exit.
watch : Continuously poll data/production/ every 30 s for new batch files.

Alerts are written to reports/alerts.log and printed to the console with
colour-coded severity (WARNING = yellow, CRITICAL = red).
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import mlflow
import pandas as pd

from monitor import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    add_predictions,
    build_datasets,
    load_batch_metadata,
    load_model_artifacts,
    load_reference,
    run_classification_report,
    run_drift_report,
    save_reports,
)

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PRODUCTION_DIR = PROJECT_ROOT / "data" / "production"
PROCESSED_PATH = PRODUCTION_DIR / "processed_batches.json"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.csv"
ALERTS_LOG_PATH = REPORTS_DIR / "alerts.log"

WATCH_INTERVAL_SECONDS = 30

# Alert thresholds
DRIFT_SHARE_WARNING = 0.15
DRIFT_SHARE_CRITICAL = 0.30
ACCURACY_DROP_THRESHOLD = 0.05

# Baseline accuracy from initial training (Phase 2)
BASELINE_ACCURACY = 0.9026

# ANSI colour codes for console output
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Processed-batch tracking ──────────────────────────────────────────────────

def load_processed_batches() -> Set[str]:
    """Load the set of already-processed batch names from JSON.

    Returns:
        Set of batch name strings (e.g. {"batch_01", "batch_02"}).
    """
    if not PROCESSED_PATH.exists():
        return set()
    try:
        with open(PROCESSED_PATH) as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read processed batches file: %s", exc)
        return set()


def save_processed_batches(processed: Set[str]) -> None:
    """Persist the set of processed batch names to JSON.

    Args:
        processed: Set of batch name strings.
    """
    try:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_PATH, "w") as f:
            json.dump(sorted(processed), f, indent=2)
    except OSError as exc:
        logger.error("Failed to save processed batches: %s", exc)


def get_unprocessed_batches(processed: Set[str]) -> List[Path]:
    """Find batch CSV files in production/ that have not been processed yet.

    Args:
        processed: Set of already-processed batch names.

    Returns:
        Sorted list of Paths for unprocessed batch files.
    """
    all_batches = sorted(PRODUCTION_DIR.glob("batch_*.csv"))
    return [p for p in all_batches if p.stem not in processed]


# ── Alerting ──────────────────────────────────────────────────────────────────

def _init_alert_log() -> None:
    """Ensure the alerts log file and its parent directory exist."""
    ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ALERTS_LOG_PATH.exists():
        ALERTS_LOG_PATH.touch()


def emit_alert(batch_name: str, level: str, reason: str) -> None:
    """Write an alert to the log file and to the console.

    Args:
        batch_name: Batch that triggered the alert.
        level: "WARNING" or "CRITICAL".
        reason: Human-readable explanation.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"{timestamp} | {batch_name} | {level} | {reason}"

    # File
    try:
        with open(ALERTS_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError as exc:
        logger.error("Failed to write alert log: %s", exc)

    # Console (coloured)
    colour = _RED if level == "CRITICAL" else _YELLOW
    logger.warning("%s[%s]%s %s — %s", colour, level, _RESET, batch_name, reason)


def check_alerts(batch_name: str, drift_metrics: Dict[str, Any], perf_metrics: Dict[str, float]) -> None:
    """Evaluate alert thresholds and emit alerts when breached.

    Args:
        batch_name: Name of the batch just processed.
        drift_metrics: Drift detection results from run_drift_report.
        perf_metrics: Classification metrics from run_classification_report.
    """
    drifted_share = drift_metrics.get("drifted_feature_share", 0.0)
    accuracy = perf_metrics.get("accuracy", 1.0)
    accuracy_drop = BASELINE_ACCURACY - accuracy

    # Drift share alerts
    if drifted_share >= DRIFT_SHARE_CRITICAL:
        emit_alert(
            batch_name, "CRITICAL",
            f"Drifted feature share {drifted_share:.1%} exceeds critical threshold ({DRIFT_SHARE_CRITICAL:.0%})",
        )
    elif drifted_share >= DRIFT_SHARE_WARNING:
        emit_alert(
            batch_name, "WARNING",
            f"Drifted feature share {drifted_share:.1%} exceeds warning threshold ({DRIFT_SHARE_WARNING:.0%})",
        )

    # Accuracy drop alert
    if accuracy_drop >= ACCURACY_DROP_THRESHOLD:
        emit_alert(
            batch_name, "CRITICAL",
            f"Accuracy {accuracy:.4f} dropped {accuracy_drop:.4f} below baseline ({BASELINE_ACCURACY:.4f})",
        )

    # Target drift alert
    if drift_metrics.get("target_drift_detected", False):
        emit_alert(
            batch_name, "WARNING",
            f"Target drift detected (p-value={drift_metrics.get('target_drift_p_value', 0):.6f})",
        )


# ── Summary CSV management ────────────────────────────────────────────────────

def append_to_summary(row: Dict[str, Any]) -> None:
    """Append a single batch result to the drift summary CSV.

    Creates the file with headers if it does not exist.

    Args:
        row: Dict with the same columns as drift_summary.csv.
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([row])
        if DRIFT_SUMMARY_PATH.exists():
            existing = pd.read_csv(DRIFT_SUMMARY_PATH)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        combined.to_csv(DRIFT_SUMMARY_PATH, index=False)
        logger.info("Drift summary updated with %s", row["batch_name"])
    except OSError as exc:
        logger.error("Failed to update drift summary: %s", exc)


# ── Core processing ──────────────────────────────────────────────────────────

def process_batch(
    batch_path: Path,
    reference: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full monitoring pipeline on a single batch and emit alerts.

    Args:
        batch_path: Path to the batch CSV.
        reference: Reference DataFrame with prediction column.
        model: Fitted sklearn model.
        preprocessor: Fitted Preprocessor instance.
        metadata: Batch metadata dict from batch_metadata.json.

    Returns:
        Summary row dict, or empty dict on failure.
    """
    batch_name = batch_path.stem
    logger.info("Processing %s ...", batch_name)

    try:
        current = pd.read_csv(batch_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", batch_path, exc)
        return {}

    try:
        current = add_predictions(current, model, preprocessor)
    except Exception as exc:
        logger.error("Preprocessing failed for %s: %s", batch_name, exc)
        return {}

    ref_ds, cur_ds = build_datasets(reference, current)

    drift_snap, drift_metrics = run_drift_report(ref_ds, cur_ds, batch_name)
    cls_snap, perf_metrics = run_classification_report(ref_ds, cur_ds, batch_name)

    save_reports(drift_snap, cls_snap, batch_name)

    # MLflow logging
    batch_meta = metadata.get(batch_name, {})
    drift_type = batch_meta.get("drift_type", "unknown")

    with mlflow.start_run(run_name=f"scheduler-{batch_name}", nested=True) as run:
        mlflow.set_tags({
            "batch_name": batch_name,
            "drift_type": drift_type,
            "stage": "scheduler",
        })
        mlflow.log_metrics({
            "dataset_drift_detected": int(drift_metrics.get("dataset_drift_detected", False)),
            "drifted_feature_share": drift_metrics.get("drifted_feature_share", 0.0),
            "drifted_feature_count": drift_metrics.get("drifted_feature_count", 0),
            "target_drift_p_value": drift_metrics.get("target_drift_p_value", 1.0),
            "target_drift_detected": int(drift_metrics.get("target_drift_detected", False)),
        })
        for col, p_val in drift_metrics.get("feature_drift_scores", {}).items():
            mlflow.log_metric(f"drift_pvalue_{col}", p_val)
        mlflow.log_metrics(perf_metrics)
        logger.info("%s — MLflow run_id: %s", batch_name, run.info.run_id)

    # Alerts
    check_alerts(batch_name, drift_metrics, perf_metrics)

    summary_row = {
        "batch_name": batch_name,
        "drift_type": drift_type,
        "dataset_drift_detected": drift_metrics.get("dataset_drift_detected", False),
        "drifted_feature_share": round(drift_metrics.get("drifted_feature_share", 0.0), 4),
        "target_drift_detected": drift_metrics.get("target_drift_detected", False),
        "target_drift_p_value": round(drift_metrics.get("target_drift_p_value", 1.0), 6),
        "accuracy": round(perf_metrics.get("accuracy", 0.0), 4),
        "precision": round(perf_metrics.get("precision", 0.0), 4),
        "recall": round(perf_metrics.get("recall", 0.0), 4),
        "f1": round(perf_metrics.get("f1", 0.0), 4),
    }

    append_to_summary(summary_row)
    return summary_row


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_once(
    reference: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
) -> None:
    """Process all unprocessed batches in one pass, then exit.

    Args:
        reference: Reference DataFrame with predictions.
        model: Fitted sklearn model.
        preprocessor: Fitted Preprocessor.
        metadata: Batch metadata dict.
    """
    processed = load_processed_batches()
    unprocessed = get_unprocessed_batches(processed)

    if not unprocessed:
        logger.info("No unprocessed batches found")
        return

    logger.info("Found %d unprocessed batch(es)", len(unprocessed))

    with mlflow.start_run(run_name="scheduler-once") as _parent:
        for batch_path in unprocessed:
            row = process_batch(batch_path, reference, model, preprocessor, metadata)
            if row:
                processed.add(batch_path.stem)
                save_processed_batches(processed)

    logger.info("Batch processing complete — %d batch(es) processed", len(unprocessed))


def run_watch(
    reference: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
) -> None:
    """Continuously poll for new batches at a fixed interval.

    Args:
        reference: Reference DataFrame with predictions.
        model: Fitted sklearn model.
        preprocessor: Fitted Preprocessor.
        metadata: Batch metadata dict.
    """
    logger.info(
        "Watch mode started — polling every %d seconds (Ctrl+C to stop)",
        WATCH_INTERVAL_SECONDS,
    )

    processed = load_processed_batches()

    try:
        while True:
            unprocessed = get_unprocessed_batches(processed)

            if unprocessed:
                logger.info("Detected %d new batch(es)", len(unprocessed))
                with mlflow.start_run(run_name="scheduler-watch") as _parent:
                    for batch_path in unprocessed:
                        row = process_batch(batch_path, reference, model, preprocessor, metadata)
                        if row:
                            processed.add(batch_path.stem)
                            save_processed_batches(processed)

            time.sleep(WATCH_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args and run the scheduler."""
    parser = argparse.ArgumentParser(
        description="Automated drift monitoring scheduler",
    )
    parser.add_argument(
        "--mode",
        choices=["once", "watch"],
        default="once",
        help="'once' processes all unprocessed batches; 'watch' polls continuously (default: once)",
    )
    args = parser.parse_args()

    logger.info("=== Phase 7: Monitoring Scheduler (mode=%s) ===", args.mode)

    _init_alert_log()

    # Load shared artifacts
    reference = load_reference()
    model, preprocessor = load_model_artifacts()
    metadata = load_batch_metadata()
    reference = add_predictions(reference, model, preprocessor)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if args.mode == "once":
        run_once(reference, model, preprocessor, metadata)
    else:
        run_watch(reference, model, preprocessor, metadata)


if __name__ == "__main__":
    main()
