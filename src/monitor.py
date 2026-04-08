"""Evidently AI monitoring — drift detection and model performance tracking.

Iterates over production batches, runs Evidently drift and classification
reports, saves HTML/JSON outputs, extracts key metrics, and logs everything
to MLflow as child runs under the baseline experiment.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd
from evidently import DataDefinition
from evidently.core.datasets import BinaryClassification, Dataset
from evidently.core.report import Report
from evidently.presets.classification import ClassificationPreset
from evidently.presets.drift import DataDriftPreset, ValueDrift

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_PATH = DATA_DIR / "reference" / "reference_data.csv"
PRODUCTION_DIR = DATA_DIR / "production"
METADATA_PATH = PRODUCTION_DIR / "batch_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "bank-marketing-baseline"

DRIFT_SHARE_THRESHOLD = 0.5  # dataset drifted if > 50% of columns drift
DRIFT_P_VALUE_THRESHOLD = 0.05  # per-column: p-value below this → drifted

NUMERICAL_COLS = [
    "age", "balance", "day_of_week", "duration",
    "campaign", "pdays", "previous",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
TARGET_COL = "y"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_reference() -> pd.DataFrame:
    """Load reference (baseline) data from CSV.

    Returns:
        Reference DataFrame with features and target.

    Raises:
        FileNotFoundError: If reference CSV is missing.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. Run src/train.py first."
        )
    df = pd.read_csv(REFERENCE_PATH)
    logger.info("Reference data loaded: %d rows", len(df))
    return df


def load_model_artifacts() -> Tuple[Any, Any]:
    """Load the trained model and preprocessor from pickle files.

    Returns:
        Tuple of (model, preprocessor).

    Raises:
        FileNotFoundError: If model or preprocessor pickle is missing.
    """
    for path, name in [(MODEL_PATH, "Model"), (PREPROCESSOR_PATH, "Preprocessor")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}. Run src/train.py first.")
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("Model and preprocessor loaded")
        return model, preprocessor
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc


def load_batch_metadata() -> Dict[str, Any]:
    """Load batch metadata JSON mapping batch names to drift types.

    Returns:
        Dict keyed by batch name (e.g. "batch_01") with drift info.
    """
    if not METADATA_PATH.exists():
        logger.warning("Batch metadata not found at %s — drift types unknown", METADATA_PATH)
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_sorted_batch_paths() -> List[Path]:
    """Return production batch CSV paths sorted by batch number.

    Returns:
        List of Path objects for each batch_XX.csv file.
    """
    paths = sorted(PRODUCTION_DIR.glob("batch_*.csv"))
    logger.info("Found %d production batches", len(paths))
    return paths


def add_predictions(
    df: pd.DataFrame,
    model: Any,
    preprocessor: Any,
) -> pd.DataFrame:
    """Add a 'prediction' column to a DataFrame using the baseline model.

    Handles dtype mismatches by coercing columns to match the reference
    schema before encoding.

    Args:
        df: Raw DataFrame with feature columns and target.
        model: Fitted sklearn model.
        preprocessor: Fitted Preprocessor from utils.

    Returns:
        DataFrame with an added 'prediction' column.
    """
    df = df.copy()
    features = df.drop(columns=[TARGET_COL], errors="ignore")

    for col in CATEGORICAL_COLS:
        if col in features.columns:
            features[col] = features[col].astype(str)

    for col in NUMERICAL_COLS:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    encoded = preprocessor.transform(features)
    df["prediction"] = model.predict(encoded)
    return df


def build_datasets(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
) -> Tuple[Dataset, Dataset]:
    """Wrap raw DataFrames into Evidently Datasets with proper schema.

    Args:
        ref: Reference DataFrame with prediction column.
        cur: Current batch DataFrame with prediction column.

    Returns:
        Tuple of (reference Dataset, current Dataset).
    """
    data_def = DataDefinition(
        numerical_columns=NUMERICAL_COLS,
        categorical_columns=CATEGORICAL_COLS + [TARGET_COL, "prediction"],
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels="prediction",
                pos_label=1,
            )
        ],
    )
    return (
        Dataset.from_pandas(ref, data_definition=data_def),
        Dataset.from_pandas(cur, data_definition=data_def),
    )


# ── Report runners ─────────────────────────────────────────────────────────────

def run_drift_report(
    ref_ds: Dataset,
    cur_ds: Dataset,
    batch_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Run DataDriftPreset + target ValueDrift and extract metrics.

    Args:
        ref_ds: Evidently reference Dataset.
        cur_ds: Evidently current Dataset.
        batch_name: Batch identifier for logging.

    Returns:
        Tuple of (Snapshot, extracted metrics dict).
    """
    report = Report([DataDriftPreset(), ValueDrift(column=TARGET_COL)])
    snap = report.run(current_data=cur_ds, reference_data=ref_ds)

    metrics: Dict[str, Any] = {
        "feature_drift_scores": {},
    }

    for _key, result in snap.metric_results.items():
        result_type = type(result).__name__
        display_name = result.display_name

        if result_type == "CountValue":
            metrics["drifted_feature_count"] = result.count.value
            metrics["drifted_feature_share"] = result.share.value
            metrics["dataset_drift_detected"] = result.share.value > DRIFT_SHARE_THRESHOLD

        elif result_type == "SingleValue" and display_name.startswith("Value drift for "):
            col_name = display_name.replace("Value drift for ", "")
            p_value = float(result.value)

            if col_name == TARGET_COL:
                metrics["target_drift_p_value"] = p_value
                metrics["target_drift_detected"] = p_value < DRIFT_P_VALUE_THRESHOLD
            else:
                metrics["feature_drift_scores"][col_name] = p_value

    logger.info(
        "%s — dataset_drift=%s, drifted_share=%.3f, target_drift=%s",
        batch_name,
        metrics.get("dataset_drift_detected"),
        metrics.get("drifted_feature_share", 0.0),
        metrics.get("target_drift_detected"),
    )
    return snap, metrics


def run_classification_report(
    ref_ds: Dataset,
    cur_ds: Dataset,
    batch_name: str,
) -> Tuple[Any, Dict[str, float]]:
    """Run ClassificationPreset and extract performance metrics.

    Args:
        ref_ds: Evidently reference Dataset.
        cur_ds: Evidently current Dataset.
        batch_name: Batch identifier for logging.

    Returns:
        Tuple of (Snapshot, dict of metric_name → float).
    """
    report = Report([ClassificationPreset()])
    snap = report.run(current_data=cur_ds, reference_data=ref_ds)

    name_map = {
        "Accuracy metric": "accuracy",
        "Precision metric": "precision",
        "Recall metric": "recall",
        "F1 score metric": "f1",
    }

    perf: Dict[str, float] = {}
    for _key, result in snap.metric_results.items():
        if hasattr(result, "value") and result.display_name in name_map:
            perf[name_map[result.display_name]] = float(result.value)

    logger.info(
        "%s — accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
        batch_name,
        perf.get("accuracy", 0),
        perf.get("precision", 0),
        perf.get("recall", 0),
        perf.get("f1", 0),
    )
    return snap, perf


# ── Save & log ─────────────────────────────────────────────────────────────────

def save_reports(
    drift_snap: Any,
    classification_snap: Any,
    batch_name: str,
) -> Tuple[Path, Path]:
    """Save Evidently drift and classification reports as HTML and JSON.

    Args:
        drift_snap: Drift report Snapshot.
        classification_snap: Classification report Snapshot.
        batch_name: Batch identifier for filenames.

    Returns:
        Tuple of (drift HTML path, drift JSON path).

    Raises:
        OSError: If files cannot be written.
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        html_path = REPORTS_DIR / f"{batch_name}_drift_report.html"
        json_path = REPORTS_DIR / f"{batch_name}_drift_report.json"
        drift_snap.save_html(str(html_path))
        drift_snap.save_json(str(json_path))

        cls_html = REPORTS_DIR / f"{batch_name}_classification_report.html"
        cls_json = REPORTS_DIR / f"{batch_name}_classification_report.json"
        classification_snap.save_html(str(cls_html))
        classification_snap.save_json(str(cls_json))

        logger.info("Reports saved for %s", batch_name)
        return html_path, json_path
    except OSError as exc:
        raise OSError(f"Failed to save reports for {batch_name}: {exc}") from exc


def log_batch_to_mlflow(
    batch_name: str,
    drift_type: str,
    drift_metrics: Dict[str, Any],
    perf_metrics: Dict[str, float],
    html_path: Path,
    json_path: Path,
) -> str:
    """Log batch monitoring results as an MLflow child run.

    Args:
        batch_name: Batch identifier (e.g. "batch_01").
        drift_type: Ground-truth drift type from metadata.
        drift_metrics: Extracted drift detection metrics.
        perf_metrics: Extracted classification performance metrics.
        html_path: Path to saved HTML report.
        json_path: Path to saved JSON report.

    Returns:
        MLflow child run ID.
    """
    with mlflow.start_run(run_name=f"monitor-{batch_name}", nested=True) as run:
        mlflow.set_tags({
            "batch_name": batch_name,
            "drift_type": drift_type,
            "stage": "monitoring",
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

        mlflow.log_artifact(str(html_path), artifact_path="drift_reports")
        mlflow.log_artifact(str(json_path), artifact_path="drift_reports")

        return run.info.run_id


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_batch(
    batch_path: Path,
    reference: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full monitoring pipeline for a single production batch.

    Args:
        batch_path: Path to the batch CSV.
        reference: Reference DataFrame (with prediction column).
        model: Fitted sklearn model.
        preprocessor: Fitted Preprocessor.
        metadata: Full batch metadata dict.

    Returns:
        Summary row dict for the drift summary CSV, or empty dict on failure.
    """
    batch_name = batch_path.stem
    logger.info("Processing %s ...", batch_name)

    try:
        current = pd.read_csv(batch_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", batch_path, exc)
        return {}

    missing_cols = set(reference.columns) - {"prediction"} - set(current.columns)
    if missing_cols:
        logger.warning("%s is missing columns %s — filling with defaults", batch_name, missing_cols)
        for col in missing_cols:
            if col in NUMERICAL_COLS:
                current[col] = 0
            else:
                current[col] = "unknown"

    current = add_predictions(current, model, preprocessor)

    ref_ds, cur_ds = build_datasets(reference, current)

    drift_snap, drift_metrics = run_drift_report(ref_ds, cur_ds, batch_name)
    cls_snap, perf_metrics = run_classification_report(ref_ds, cur_ds, batch_name)

    html_path, json_path = save_reports(drift_snap, cls_snap, batch_name)

    batch_meta = metadata.get(batch_name, {})
    drift_type = batch_meta.get("drift_type", "unknown")

    run_id = log_batch_to_mlflow(
        batch_name, drift_type, drift_metrics, perf_metrics, html_path, json_path,
    )
    logger.info("%s — MLflow run_id: %s", batch_name, run_id)

    return {
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


def save_summary(rows: List[Dict[str, Any]]) -> Path:
    """Save the monitoring summary as a CSV file.

    Args:
        rows: List of per-batch summary dicts.

    Returns:
        Path to the saved CSV.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "drift_summary.csv"
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        logger.info("Drift summary saved to %s", summary_path)
        return summary_path
    except OSError as exc:
        raise OSError(f"Failed to save drift summary: {exc}") from exc


def main() -> None:
    """Run the full monitoring pipeline across all production batches."""
    logger.info("=== Phase 4: Evidently AI Monitoring ===")

    reference = load_reference()
    model, preprocessor = load_model_artifacts()
    metadata = load_batch_metadata()

    reference = add_predictions(reference, model, preprocessor)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    batch_paths = get_sorted_batch_paths()

    summary_rows: List[Dict[str, Any]] = []
    with mlflow.start_run(run_name="monitoring-sweep") as _parent_run:
        for batch_path in batch_paths:
            row = process_batch(batch_path, reference, model, preprocessor, metadata)
            if row:
                summary_rows.append(row)

    save_summary(summary_rows)

    logger.info("=== Monitoring complete — %d batches processed ===", len(summary_rows))
    logger.info("Summary:\n%s", pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
