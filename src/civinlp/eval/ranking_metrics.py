"""
Ranking metrics for CiviNLP Triage.

Provides Top-K Recall, NDCG@K, and Spearman rank correlation
for evaluating prioritization quality.
"""

import logging
from typing import List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def top_k_recall(
    ranked_ids: List[str],
    critical_ids: set,
    k: int,
) -> float:
    """
    Compute Top-K Recall: fraction of critical items captured in the top K positions.

    Args:
        ranked_ids: Ordered list of item IDs (highest priority first).
        critical_ids: Set of IDs considered critical (ground truth).
        k: Number of top positions to consider.

    Returns:
        Fraction of critical items in top-K (0.0 to 1.0).
    """
    if not critical_ids:
        return 0.0

    top_k = set(ranked_ids[:k])
    captured = top_k & critical_ids
    recall = len(captured) / len(critical_ids)

    logger.debug(
        f"Top-{k} recall: {recall:.4f} ({len(captured)}/{len(critical_ids)} critical)"
    )
    return float(recall)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    Args:
        relevance_scores: Ordered list of relevance scores (at each rank position).
        k: Number of positions to consider.

    Returns:
        DCG@K value.
    """
    scores = np.array(relevance_scores[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum(scores / discounts))


def ndcg_at_k(
    ranked_ids: List[str],
    relevance_map: dict,
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        ranked_ids: Ordered list of item IDs (highest priority first).
        relevance_map: Dict mapping item_id -> relevance score (e.g., severity).
        k: Number of positions to consider.

    Returns:
        NDCG@K value (0.0 to 1.0). Returns 0.0 if ideal DCG is 0.
    """
    # Actual DCG from the ranked list
    actual_relevance = [relevance_map.get(item_id, 0.0) for item_id in ranked_ids[:k]]
    actual_dcg = dcg_at_k(actual_relevance, k)

    # Ideal DCG (sort all relevance scores descending)
    ideal_relevance = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0

    ndcg = actual_dcg / ideal_dcg
    logger.debug(f"NDCG@{k}: {ndcg:.4f}")
    return float(ndcg)


def spearman_correlation(
    ranking_a: List[str],
    ranking_b: List[str],
) -> float:
    """
    Compute Spearman rank correlation between two rankings.

    Items must overlap. Only shared items are compared.

    Args:
        ranking_a: First ordered list of item IDs.
        ranking_b: Second ordered list of item IDs.

    Returns:
        Spearman correlation coefficient (-1.0 to 1.0).
    """
    shared = set(ranking_a) & set(ranking_b)
    if len(shared) < 2:
        return 0.0

    rank_a = {item: i for i, item in enumerate(ranking_a) if item in shared}
    rank_b = {item: i for i, item in enumerate(ranking_b) if item in shared}

    items = sorted(shared)
    ranks_a = [rank_a[item] for item in items]
    ranks_b = [rank_b[item] for item in items]

    corr, _ = stats.spearmanr(ranks_a, ranks_b)
    logger.debug(f"Spearman correlation: {corr:.4f} over {len(shared)} shared items")
    return float(corr)


def top_k_recall_curve(
    ranked_ids: List[str],
    critical_ids: set,
    k_values: Optional[List[int]] = None,
) -> dict:
    """
    Compute Top-K recall for multiple values of K.

    Args:
        ranked_ids: Ordered list of item IDs.
        critical_ids: Set of critical item IDs.
        k_values: List of K values. Defaults to [5, 10, 20, 50].

    Returns:
        Dict mapping K -> recall value.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    return {k: top_k_recall(ranked_ids, critical_ids, k) for k in k_values}
