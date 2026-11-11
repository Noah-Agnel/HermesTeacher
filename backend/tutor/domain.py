from collections.abc import Mapping
from math import isfinite


def update_level(current: float, result: float, eta: float, aided: bool = False) -> float:
    """Update one mastery score, unless the attempt used help.

    ``current`` is k, ``result`` is r, and ``eta`` controls how much the
    latest result moves the score. All three numeric inputs use the [0, 1]
    scale so the returned level stays on that same scale.
    """
    # Validate every score and the learning rate before doing any arithmetic.
    # The range check also rejects NaN, because NaN is not between 0 and 1.
    for name, value in (("current", current), ("result", result), ("eta", eta)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    # An assisted answer is not evidence of independent mastery. Keep k intact.
    if aided:
        return current

    # Move a fraction eta of the distance from the old level to the result.
    # eta=0 leaves the score unchanged; eta=1 adopts the result completely.
    return current + eta * (result - current)


def weighted_gap(
    current: Mapping[str, float],
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the weighted mean positive shortfall from target levels.

    Current and target maps must describe the same concepts. If weights are
    omitted, every concept has equal weight. The result is in [0, 1].
    """
    # Matching keys ensure every current score has exactly one target score.
    if current.keys() != target.keys():
        raise ValueError("current and target must contain the same concepts")
    concepts = current.keys()

    # Validate both sets of levels before computing any concept-level gaps.
    for name, levels in (("current", current), ("target", target)):
        for concept, value in levels.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name}[{concept!r}] must be between 0 and 1")

    # A missing weight map means a uniform weighting of all concepts.
    if weights is None:
        weights = {concept: 1.0 for concept in concepts}
    # Supplied weights must align with the same concepts, avoiding silent drops.
    elif weights.keys() != concepts:
        raise ValueError("weights must contain exactly the same concepts")

    # Weights may be any finite, non-negative numbers; their scale cancels in
    # the weighted mean, so they do not need to sum to one.
    for concept, weight in weights.items():
        if weight < 0 or not isfinite(weight):
            raise ValueError(f"weights[{concept!r}] must be finite and non-negative")

    # If all weights are zero, there is no weighted shortfall to report.
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    # Positive differences are unmet targets. max(..., 0) gives zero to any
    # concept already at or above target; its weight remains in the overall
    # mean denominator so fully met concepts contribute no shortfall.
    return (
        sum(
            weights[concept] * max(target[concept] - current[concept], 0.0)
            for concept in concepts
        )
        / total_weight
    )


def progression(initial_gap: float, current_gap: float) -> float:
    """Return the clamped fraction of the initial gap that has been closed.

    An initial gap of zero is treated as complete (1.0) if the current gap is
    also zero; if a gap appears later, progression is 0.0.
    """
    # Gaps are normalized scores, so both must be in the same [0, 1] range.
    for name, value in (("initial_gap", initial_gap), ("current_gap", current_gap)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    # Avoid dividing by zero: no initial gap means nothing was left to close.
    if initial_gap == 0:
        return 1.0 if current_gap == 0 else 0.0

    # Fraction closed = (initial - current) / initial. Clamp it so a growing
    # gap cannot produce negative progress and overshoot cannot exceed 100%.
    return min(1.0, max(0.0, (initial_gap - current_gap) / initial_gap))
