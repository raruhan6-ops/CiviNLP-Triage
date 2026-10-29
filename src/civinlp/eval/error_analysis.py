from typing import List

import pandas as pd


def get_top_errors(
    texts: List[str],
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Identify the top K errors, prioritizing high-confidence errors if confidence provided.
    """
    df = pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred})

    if confidences is not None:
        df["confidence"] = confidences

    # Filter for errors
    errors = df[df["y_true"] != df["y_pred"]].copy()

    if confidences is not None:
        # Sort by confidence descending (High confidence wrong answers)
        errors = errors.sort_values("confidence", ascending=False)

    return errors.head(k)


def generate_error_report(results_df: pd.DataFrame, output_path: str):
    """
    Save error analysis to CSV.
    """
    results_df.to_csv(output_path, index=False)
