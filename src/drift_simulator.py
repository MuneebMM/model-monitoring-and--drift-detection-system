"""Drift simulation engine — generates production batches with controlled drift.

Batch schedule
--------------
Batches 01-02 : no_drift     — clean sample from reference with minor noise
Batches 03-07 : gradual_drift — progressive numerical shifts over 5 steps
Batch  08     : sudden_drift  — abrupt demographic + balance shift
Batches 09-10 : concept_drift — features stable, label relationship changes
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
REFERENCE_PATH = DATA_DIR / "reference" / "reference_data.csv"
PRODUCTION_DIR = DATA_DIR / "production"
METADATA_PATH = PRODUCTION_DIR / "batch_metadata.json"

BATCH_SIZE = 1000
RANDOM_STATE = 42

TARGET_COL = "y"
NUMERICAL_COLS = [
    "age", "balance", "day_of_week", "duration",
    "campaign", "pdays", "previous",
]

# Per-step additive shifts for gradual drift (applied cumulatively each batch)
GRADUAL_SHIFTS = {
    "age": 2.0,        # +2 years per step  → +10 by step 5
    "balance": 350.0,  # +€350 per step     → +1750 by step 5
    "duration": 35.0,  # +35 s per step     → +175 s by step 5
    "campaign": 0.6,   # +0.6 calls/step    → +3 by step 5
}

# Sudden drift parameters
SUDDEN_AGE_THRESHOLD = 48         # keep only older customers
SUDDEN_BALANCE_MULTIPLIER = 3.0   # triple their balances
SUDDEN_RETIRED_FRACTION = 0.65    # fraction of jobs reassigned to "retired"

# Concept drift: subgroup that flips label
# High-balance customers who originally declined now accept
CONCEPT_BALANCE_THRESHOLD = 1500  # balance > threshold
CONCEPT_FLIP_FRACTION = 0.75      # fraction of that subgroup to flip 0→1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_reference() -> pd.DataFrame:
    """Load reference (baseline) data from CSV.

    Returns:
        Reference DataFrame with all original columns.

    Raises:
        FileNotFoundError: If reference CSV does not exist.
        RuntimeError: If loading fails.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. "
            "Run src/train.py first to generate it."
        )
    try:
        df = pd.read_csv(REFERENCE_PATH)
        logger.info("Reference data loaded: %d rows, %d cols", *df.shape)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load reference data: {exc}") from exc


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n rows with replacement using a seeded RNG.

    Args:
        df: Source DataFrame.
        n: Number of rows to sample.
        rng: NumPy random generator for reproducibility.

    Returns:
        Sampled DataFrame with reset index.
    """
    idx = rng.integers(0, len(df), size=n)
    return df.iloc[idx].copy().reset_index(drop=True)


def no_drift(
    reference: pd.DataFrame,
    batch_index: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a batch that mirrors the reference distribution.

    Adds tiny Gaussian noise (σ = 1% of each feature's std) to numerical
    columns so rows are not exact duplicates of training data.

    Args:
        reference: Full reference DataFrame.
        batch_index: 1-based batch number (used to seed the RNG).

    Returns:
        Tuple of (batch DataFrame, params dict for metadata).
    """
    rng = np.random.default_rng(RANDOM_STATE + batch_index)
    batch = _sample(reference, BATCH_SIZE, rng)

    noise_scale = 0.01
    for col in NUMERICAL_COLS:
        col_std = reference[col].std()
        noise = rng.normal(0, noise_scale * col_std, size=BATCH_SIZE)
        batch[col] = (batch[col] + noise).round().astype(int)

    params = {"noise_scale": noise_scale}
    logger.info("Batch %02d [no_drift] — %d rows, minor noise only", batch_index, BATCH_SIZE)
    return batch, params


def gradual_drift(
    reference: pd.DataFrame,
    batch_index: int,
    step: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a batch with progressively increasing numerical shift.

    Each step applies GRADUAL_SHIFTS additively, so by step 5 the means
    have drifted by 5× the per-step delta — simulating slow distribution
    creep that a static model may not immediately flag.

    Args:
        reference: Full reference DataFrame.
        batch_index: 1-based batch number (used to seed the RNG).
        step: Drift step 1–5 (step 1 = smallest shift).

    Returns:
        Tuple of (batch DataFrame, params dict for metadata).
    """
    rng = np.random.default_rng(RANDOM_STATE + batch_index)
    batch = _sample(reference, BATCH_SIZE, rng)

    applied_shifts: Dict[str, float] = {}
    for col, per_step_delta in GRADUAL_SHIFTS.items():
        total_shift = per_step_delta * step
        batch[col] = (batch[col] + total_shift).round().astype(int)
        applied_shifts[col] = total_shift

    params = {"step": step, "cumulative_shifts": applied_shifts}
    logger.info(
        "Batch %02d [gradual_drift] step=%d — age+%.0f, balance+%.0f, "
        "duration+%.0f, campaign+%.1f",
        batch_index, step,
        applied_shifts["age"], applied_shifts["balance"],
        applied_shifts["duration"], applied_shifts["campaign"],
    )
    return batch, params


def sudden_drift(
    reference: pd.DataFrame,
    batch_index: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a batch with abrupt demographic and balance shift.

    Strategy:
    - Restrict to customers aged > SUDDEN_AGE_THRESHOLD (older segment).
    - Multiply balance by SUDDEN_BALANCE_MULTIPLIER (wealthier cohort).
    - Reassign SUDDEN_RETIRED_FRACTION of jobs to "retired".

    This simulates a sales campaign that was suddenly redirected at a
    completely different customer segment.

    Args:
        reference: Full reference DataFrame.
        batch_index: 1-based batch number.

    Returns:
        Tuple of (batch DataFrame, params dict for metadata).
    """
    rng = np.random.default_rng(RANDOM_STATE + batch_index)

    older = reference[reference["age"] > SUDDEN_AGE_THRESHOLD]
    logger.info(
        "Batch %02d [sudden_drift] — %d rows available with age > %d",
        batch_index, len(older), SUDDEN_AGE_THRESHOLD,
    )

    batch = _sample(older, BATCH_SIZE, rng)
    batch["balance"] = (batch["balance"] * SUDDEN_BALANCE_MULTIPLIER).round().astype(int)

    retire_mask = rng.random(BATCH_SIZE) < SUDDEN_RETIRED_FRACTION
    batch.loc[retire_mask, "job"] = "retired"

    params = {
        "age_threshold": SUDDEN_AGE_THRESHOLD,
        "balance_multiplier": SUDDEN_BALANCE_MULTIPLIER,
        "retired_fraction": SUDDEN_RETIRED_FRACTION,
    }
    logger.info(
        "Batch %02d [sudden_drift] — %d rows, balance ×%.0f, %.0f%% jobs→retired",
        batch_index, BATCH_SIZE,
        SUDDEN_BALANCE_MULTIPLIER, SUDDEN_RETIRED_FRACTION * 100,
    )
    return batch, params


def concept_drift(
    reference: pd.DataFrame,
    batch_index: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a batch where the feature→label relationship has changed.

    Feature distributions remain close to the reference, but high-balance
    customers who originally declined (y=0) are now flipped to accept (y=1).
    This simulates a scenario where improved interest rates have changed the
    actual customer decision pattern — the model's learned boundary is now wrong.

    Args:
        reference: Full reference DataFrame.
        batch_index: 1-based batch number.

    Returns:
        Tuple of (batch DataFrame, params dict for metadata).
    """
    rng = np.random.default_rng(RANDOM_STATE + batch_index)
    batch = _sample(reference, BATCH_SIZE, rng)

    # Subgroup: high balance AND originally declined
    subgroup_mask = (batch["balance"] > CONCEPT_BALANCE_THRESHOLD) & (batch[TARGET_COL] == 0)
    subgroup_idx = batch.index[subgroup_mask]

    n_eligible = len(subgroup_idx)
    n_to_flip = int(n_eligible * CONCEPT_FLIP_FRACTION)
    flip_idx = rng.choice(subgroup_idx, size=n_to_flip, replace=False)

    batch.loc[flip_idx, TARGET_COL] = 1

    original_positive_rate = reference[TARGET_COL].mean()
    new_positive_rate = batch[TARGET_COL].mean()

    params = {
        "balance_threshold": CONCEPT_BALANCE_THRESHOLD,
        "flip_fraction": CONCEPT_FLIP_FRACTION,
        "n_eligible_in_batch": n_eligible,
        "n_flipped": n_to_flip,
        "original_positive_rate": round(original_positive_rate, 4),
        "batch_positive_rate": round(new_positive_rate, 4),
    }
    logger.info(
        "Batch %02d [concept_drift] — %d/%d high-balance negatives flipped; "
        "positive rate %.3f → %.3f",
        batch_index, n_to_flip, n_eligible,
        original_positive_rate, new_positive_rate,
    )
    return batch, params


def save_batch(batch: pd.DataFrame, batch_index: int) -> Path:
    """Write a batch DataFrame to data/production/batch_XX.csv.

    Args:
        batch: Batch DataFrame to persist.
        batch_index: 1-based batch number (zero-padded to 2 digits).

    Returns:
        Path where the batch was saved.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        path = PRODUCTION_DIR / f"batch_{batch_index:02d}.csv"
        batch.to_csv(path, index=False)
        logger.debug("Saved %s", path)
        return path
    except OSError as exc:
        raise OSError(f"Failed to save batch {batch_index:02d}: {exc}") from exc


def save_metadata(metadata: Dict[str, Any]) -> None:
    """Persist batch metadata mapping to JSON.

    Args:
        metadata: Dict mapping batch name to drift type and parameters.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Batch metadata saved to %s", METADATA_PATH)
    except OSError as exc:
        raise OSError(f"Failed to save metadata: {exc}") from exc


def main() -> None:
    """Generate all 10 production batches according to the drift schedule.

    Schedule
    --------
    Batches 01-02 : no_drift
    Batches 03-07 : gradual_drift (steps 1-5)
    Batch  08     : sudden_drift
    Batches 09-10 : concept_drift
    """
    logger.info("=== Phase 3: Drift Simulation Engine ===")

    reference = load_reference()
    metadata: Dict[str, Any] = {}

    # ── Batches 01-02: no drift ────────────────────────────────────────────────
    for batch_idx in [1, 2]:
        batch, params = no_drift(reference, batch_idx)
        save_batch(batch, batch_idx)
        metadata[f"batch_{batch_idx:02d}"] = {"drift_type": "no_drift", "params": params}

    # ── Batches 03-07: gradual drift (step 1 → step 5) ────────────────────────
    for step, batch_idx in enumerate(range(3, 8), start=1):
        batch, params = gradual_drift(reference, batch_idx, step)
        save_batch(batch, batch_idx)
        metadata[f"batch_{batch_idx:02d}"] = {"drift_type": "gradual_drift", "params": params}

    # ── Batch 08: sudden drift ─────────────────────────────────────────────────
    batch, params = sudden_drift(reference, 8)
    save_batch(batch, 8)
    metadata["batch_08"] = {"drift_type": "sudden_drift", "params": params}

    # ── Batches 09-10: concept drift ──────────────────────────────────────────
    for batch_idx in [9, 10]:
        batch, params = concept_drift(reference, batch_idx)
        save_batch(batch, batch_idx)
        metadata[f"batch_{batch_idx:02d}"] = {"drift_type": "concept_drift", "params": params}

    save_metadata(metadata)
    logger.info(
        "=== Simulation complete — %d batches written to %s ===",
        len(metadata), PRODUCTION_DIR,
    )


if __name__ == "__main__":
    main()
