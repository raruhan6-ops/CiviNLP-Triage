"""
Data splitting utilities for CiviNLP Triage.

Supports time-based and area-based splitting strategies for
both road and bridge datasets. Ensures no data leakage between splits.
"""

import pandas as pd
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    time_column: str = "created_at",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically by a time column.

    The earliest `train_frac` of rows form the training set,
    the next `val_frac` form validation, and the rest form test.

    Args:
        df: Input DataFrame.
        time_column: Name of the datetime column to sort by.
        train_frac: Fraction for training (default 0.7).
        val_frac: Fraction for validation (default 0.1).
        test_frac: Fraction for testing (default 0.2).

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If fractions don't sum to ~1.0 or time_column is missing.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    # Sort by time
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    logger.info(
        f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def area_based_split(
    df: pd.DataFrame,
    area_column: str = "area_id",
    test_areas: Optional[List[str]] = None,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by area_id to test generalization across zones.

    If `test_areas` is provided, those areas are held out for test.
    Otherwise, randomly selects areas to fill `test_frac` of the data.

    Args:
        df: Input DataFrame.
        area_column: Name of the area column.
        test_areas: Explicit list of area_ids to hold out for test.
        test_frac: Approximate fraction of data to reserve for test if test_areas not given.
        val_frac: Fraction of ALL rows to use for validation, drawn from the
            non-test areas (capped so at least one training row remains).
        seed: Random seed for area selection.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if area_column not in df.columns:
        raise ValueError(f"Column '{area_column}' not found in DataFrame")

    all_areas = df[area_column].unique()

    if test_areas is None:
        # Randomly select areas until we hit ~test_frac of total rows
        import numpy as np
        rng = np.random.RandomState(seed)
        shuffled_areas = rng.permutation(all_areas)

        test_areas = []
        test_count = 0
        target_count = int(len(df) * test_frac)

        for area in shuffled_areas:
            area_count = len(df[df[area_column] == area])
            if test_count + area_count <= target_count * 1.2:  # Allow slight overshoot
                test_areas.append(area)
                test_count += area_count
            if test_count >= target_count:
                break

    test_mask = df[area_column].isin(test_areas)
    test_df = df[test_mask].copy()
    remaining = df[~test_mask].copy()

    # Split remaining into train/val
    # Based on total rows so it stays correct when test_areas is given explicitly
    val_size = int(round(len(df) * val_frac))
    val_size = max(0, min(val_size, len(remaining) - 1))

    val_df = remaining.sample(n=val_size, random_state=seed)
    train_df = remaining.drop(val_df.index)

    logger.info(
        f"Area-based split: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)} (test_areas={test_areas})"
    )
    return train_df, val_df, test_df


def group_split(
    df: pd.DataFrame,
    group_column: str,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by whole groups: `test_frac` / `val_frac` are fractions of GROUPS,
    so every row of a group lands in exactly one split.

    Use for bridges (group_column="bridge_id", so repeated inspections of one
    bridge never leak across splits) or for zones (group_column="area_id").
    """
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in DataFrame")
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be < 1.0")

    import numpy as np
    rng = np.random.RandomState(seed)
    groups = rng.permutation(sorted(df[group_column].astype(str).unique()))
    n_test = int(round(len(groups) * test_frac))
    n_val = int(round(len(groups) * val_frac))
    if len(groups) - n_test - n_val < 1:
        raise ValueError(f"Too few groups ({len(groups)}) for the requested split")

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    keys = df[group_column].astype(str)
    test_df = df[keys.isin(test_groups)].copy()
    val_df = df[keys.isin(val_groups)].copy()
    train_df = df[~keys.isin(test_groups | val_groups)].copy()

    logger.info(
        f"Group split on '{group_column}': train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)} ({n_test} test / {n_val} val groups)"
    )
    return train_df, val_df, test_df


def bridge_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 42):
    """Split bridge records by bridge_id (see configs/bridges.yaml)."""
    return group_split(df, "bridge_id", test_frac=test_frac, val_frac=val_frac, seed=seed)
