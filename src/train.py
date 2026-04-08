"""Baseline model training with MLflow experiment tracking."""

import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import load_and_preprocess

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

MLFLOW_TRACKING_URI = f"sqlite:///{Path(__file__).parent.parent / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "bank-marketing-baseline"
RF_MAX_DEPTH = 10
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(
    X_train,
    y_train,
) -> RandomForestClassifier:
    """Instantiate and fit a RandomForestClassifier on training data.

    Args:
        X_train: Encoded training feature DataFrame.
        y_train: Binary training labels.

    Returns:
        Fitted RandomForestClassifier.

    Raises:
        RuntimeError: If model training fails.
    """
    try:
        logger.info(
            "Training RandomForestClassifier (n_estimators=%d, max_depth=%d)...",
            RF_N_ESTIMATORS,
            RF_MAX_DEPTH,
        )
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete")
        return model
    except Exception as exc:
        raise RuntimeError(f"Model training failed: {exc}") from exc


def evaluate_model(model: RandomForestClassifier, X_test, y_test) -> dict:
    """Compute classification metrics on the test set.

    Args:
        model: Fitted RandomForestClassifier.
        X_test: Encoded test feature DataFrame.
        y_test: Binary test labels.

    Returns:
        Dict mapping metric name to float value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    for name, value in metrics.items():
        logger.info("  %-12s %.4f", name, value)

    return metrics


def save_model(model: RandomForestClassifier, preprocessor) -> None:
    """Pickle the trained model and preprocessor to the models/ directory.

    Args:
        model: Fitted RandomForestClassifier.
        preprocessor: Fitted Preprocessor instance from utils.

    Raises:
        OSError: If files cannot be written.
    """
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", MODEL_PATH)

        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info("Preprocessor saved to %s", PREPROCESSOR_PATH)
    except OSError as exc:
        raise OSError(f"Failed to save model artifacts: {exc}") from exc


def log_to_mlflow(model: RandomForestClassifier, metrics: dict) -> str:
    """Log parameters, metrics, and model artifact to MLflow.

    Args:
        model: Fitted RandomForestClassifier.
        metrics: Dict of metric name → float value.

    Returns:
        MLflow run ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "random_state": RF_RANDOM_STATE,
        })

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="bank-marketing-rf-baseline",
        )

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="pickles")
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="pickles")

        run_id = run.info.run_id
        logger.info("MLflow run logged — run_id: %s", run_id)
        return run_id


def main() -> None:
    """Full training pipeline: load data → train → evaluate → log → save."""
    logger.info("=== Phase 2: Baseline Model Training ===")

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess()

    model = train_model(X_train, y_train)

    logger.info("Evaluating on test set:")
    metrics = evaluate_model(model, X_test, y_test)

    save_model(model, preprocessor)

    run_id = log_to_mlflow(model, metrics)
    logger.info("Training pipeline complete. MLflow run_id: %s", run_id)


if __name__ == "__main__":
    main()
