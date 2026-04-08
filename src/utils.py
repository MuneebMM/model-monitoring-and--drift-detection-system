"""Shared data loading, preprocessing, and splitting utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_ID = 222
TEST_SIZE = 0.2
RANDOM_STATE = 42

DATA_DIR = Path(__file__).parent.parent / "data"
REFERENCE_PATH = DATA_DIR / "reference" / "reference_data.csv"
TEST_DATA_PATH = DATA_DIR / "raw" / "test_data.csv"

NUMERICAL_COLS = [
    "age", "balance", "day_of_week", "duration",
    "campaign", "pdays", "previous",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]
TARGET_COL = "y"

logger = logging.getLogger(__name__)


@dataclass
class Preprocessor:
    """Holds fitted encoders and scaler for reproducible transforms.

    Attributes:
        label_encoders: Mapping of column name to its fitted LabelEncoder.
        scaler: Fitted StandardScaler for numerical columns.
        numerical_cols: Names of numerical feature columns.
        categorical_cols: Names of categorical feature columns.
    """

    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    numerical_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Fit encoders and scaler on training features.

        Args:
            X: Raw feature DataFrame (train split only).

        Returns:
            self, for chaining.
        """
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            logger.debug("Fitted LabelEncoder for column '%s'", col)

        self.scaler.fit(X[self.numerical_cols])
        logger.debug("Fitted StandardScaler on %d numerical columns", len(self.numerical_cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders and scaler to a feature DataFrame.

        Args:
            X: Raw feature DataFrame.

        Returns:
            Transformed copy of X.
        """
        X = X.copy()
        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))
        X[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform in one step.

        Args:
            X: Raw feature DataFrame (train split only).

        Returns:
            Transformed copy of X.
        """
        return self.fit(X).transform(X)


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch the Bank Marketing dataset from UCI repository.

    Returns:
        Tuple of (features DataFrame, target Series with binary int labels).

    Raises:
        RuntimeError: If the dataset cannot be fetched.
    """
    try:
        logger.info("Fetching UCI Bank Marketing dataset (id=%d)...", DATASET_ID)
        dataset = fetch_ucirepo(id=DATASET_ID)
        X = dataset.data.features
        y = dataset.data.targets.iloc[:, 0].map({"yes": 1, "no": 0})
        logger.info("Dataset loaded: %d rows, %d features", X.shape[0], X.shape[1])
        return X, y
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch dataset (id={DATASET_ID}): {exc}") from exc


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Args:
        X: Feature DataFrame.
        y: Binary target Series.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split complete — train: %d rows, test: %d rows", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist raw (pre-encoding) train and test splits to CSV.

    The train split is saved as the drift reference baseline.
    Evidently works best with unencoded data, so raw values are saved.

    Args:
        X_train: Raw training features.
        X_test: Raw test features.
        y_train: Training labels.
        y_test: Test labels.

    Raises:
        OSError: If the CSV files cannot be written.
    """
    try:
        reference_df = X_train.copy()
        reference_df[TARGET_COL] = y_train.values
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        reference_df.to_csv(REFERENCE_PATH, index=False)
        logger.info("Reference data saved to %s", REFERENCE_PATH)

        test_df = X_test.copy()
        test_df[TARGET_COL] = y_test.values
        TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info("Test data saved to %s", TEST_DATA_PATH)
    except OSError as exc:
        raise OSError(f"Failed to save data splits: {exc}") from exc


def load_and_preprocess() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, "Preprocessor"
]:
    """End-to-end data pipeline: load, split, save CSVs, preprocess.

    Saves raw splits before encoding so Evidently can use unencoded
    reference data for drift comparison in later phases.

    Returns:
        Tuple of (X_train_enc, X_test_enc, y_train, y_test, preprocessor).
    """
    X, y = load_dataset()
    X_train_raw, X_test_raw, y_train, y_test = split_data(X, y)

    save_splits(X_train_raw, X_test_raw, y_train, y_test)

    preprocessor = Preprocessor(
        numerical_cols=NUMERICAL_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )
    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_test_enc = preprocessor.transform(X_test_raw)
    logger.info("Preprocessing complete")

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor
