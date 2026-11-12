from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from math import isfinite


DEFAULT_PREREQUISITE_THRESHOLD = 0.75


class AttemptType(str, Enum):
    """Kinds of scored attempts; categories currently share one update rule."""

    EXERCISE = "exercise"
    EVALUATION = "evaluation"
    FINAL_EXAM = "final_exam"


@dataclass(frozen=True)
class Concept:
    """A concept and its prerequisite edges and learning priorities."""

    id: str
    prerequisites: frozenset[str] = frozenset()
    target_level: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("concept id must be a non-empty string")
        prerequisites = frozenset(self.prerequisites)
        if any(not isinstance(item, str) or not item.strip() for item in prerequisites):
            raise ValueError("prerequisite ids must be non-empty strings")
        object.__setattr__(self, "prerequisites", prerequisites)
        if not 0 <= self.target_level <= 1:
            raise ValueError("target_level must be between 0 and 1")
        if self.weight < 0 or not isfinite(self.weight):
            raise ValueError("weight must be finite and non-negative")


@dataclass(frozen=True)
class Attempt:
    """A scored response with its category and assistance assessment."""

    kind: AttemptType
    score: float
    assistance_coefficient: float = 1.0
    difficulty_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AttemptType):
            raise ValueError("kind must be an AttemptType")
        for name, value in (
            ("score", self.score),
            ("assistance_coefficient", self.assistance_coefficient),
            ("difficulty_coefficient", self.difficulty_coefficient),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


def topological_order(concepts: Mapping[str, Concept]) -> tuple[str, ...]:
    """Validate a concept graph and return a deterministic topological order."""
    for concept_id, concept in concepts.items():
        if not isinstance(concept, Concept):
            raise ValueError(f"concepts[{concept_id!r}] must be a Concept")
        if concept_id != concept.id:
            raise ValueError("each concept mapping key must match its concept id")
        for prerequisite in concept.prerequisites:
            if prerequisite not in concepts:
                raise ValueError(
                    f"concept {concept_id!r} has unknown prerequisite {prerequisite!r}"
                )

    indegree = {concept_id: len(concept.prerequisites) for concept_id, concept in concepts.items()}
    children = {concept_id: [] for concept_id in concepts}
    for concept_id, concept in concepts.items():
        for prerequisite in concept.prerequisites:
            children[prerequisite].append(concept_id)

    ready = []
    for concept_id, degree in indegree.items():
        if degree == 0:
            heappush(ready, concept_id)

    ordered = []
    while ready:
        concept_id = heappop(ready)
        ordered.append(concept_id)
        for child in sorted(children[concept_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(ready, child)

    if len(ordered) != len(concepts):
        remaining = sorted(set(concepts) - set(ordered))
        raise ValueError(f"concept graph is cyclic; unresolved concepts: {remaining!r}")
    return tuple(ordered)


def plan_concepts(
    concepts: Mapping[str, Concept],
    current_levels: Mapping[str, float],
    prerequisite_threshold: float = DEFAULT_PREREQUISITE_THRESHOLD,
) -> tuple[str, ...]:
    """Return concepts to work on in prerequisite-safe priority order.

    Concepts below target are included. Any prerequisite below the required
    threshold is added even when it is already at its own target. Among the
    currently available concepts, higher weight wins, then larger shortfall,
    then lexicographically smaller id for deterministic ties.
    """
    topological_order(concepts)
    if current_levels.keys() != concepts.keys():
        raise ValueError("current_levels must contain exactly the concept ids")
    if not 0 <= prerequisite_threshold <= 1:
        raise ValueError("prerequisite_threshold must be between 0 and 1")
    for concept_id, level in current_levels.items():
        if not 0 <= level <= 1:
            raise ValueError(f"current_levels[{concept_id!r}] must be between 0 and 1")

    selected = {
        concept_id
        for concept_id, concept in concepts.items()
        if concept.target_level > current_levels[concept_id]
    }
    required_prerequisites = set()
    visited = set()
    pending = list(sorted(selected))

    while pending:
        concept_id = pending.pop()
        if concept_id in visited:
            continue
        visited.add(concept_id)
        for prerequisite in concepts[concept_id].prerequisites:
            if current_levels[prerequisite] < prerequisite_threshold:
                required_prerequisites.add(prerequisite)
                selected.add(prerequisite)
                if prerequisite not in visited:
                    pending.append(prerequisite)

    if not selected:
        return ()

    priority_gaps = {}
    for concept_id in selected:
        concept = concepts[concept_id]
        target_shortfall = max(concept.target_level - current_levels[concept_id], 0.0)
        threshold_shortfall = (
            max(prerequisite_threshold - current_levels[concept_id], 0.0)
            if concept_id in required_prerequisites
            else 0.0
        )
        priority_gaps[concept_id] = max(target_shortfall, threshold_shortfall)

    indegree = {
        concept_id: sum(
            prerequisite in selected
            for prerequisite in concepts[concept_id].prerequisites
        )
        for concept_id in selected
    }
    children = {concept_id: [] for concept_id in selected}
    for concept_id in selected:
        for prerequisite in concepts[concept_id].prerequisites:
            if prerequisite in selected:
                children[prerequisite].append(concept_id)

    ready = []
    for concept_id, degree in indegree.items():
        if degree == 0:
            heappush(
                ready,
                (-concepts[concept_id].weight, -priority_gaps[concept_id], concept_id),
            )

    ordered = []
    while ready:
        _, _, concept_id = heappop(ready)
        ordered.append(concept_id)
        for child in sorted(children[concept_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(
                    ready,
                    (-concepts[child].weight, -priority_gaps[child], child),
                )

    return tuple(ordered)


def update_level(
    current: float,
    result: float,
    eta: float,
    assistance_coefficient: float = 1.0,
    difficulty_coefficient: float = 1.0,
) -> float:
    """Update mastery, scaling the learning rate for help and difficulty.

    ``current`` is k, ``result`` is r, and ``eta`` is the base learning rate.
    Both coefficients are in [0, 1]. A help coefficient of 0 produces no
    mastery update; a larger difficulty coefficient gives the scored exercise
    more influence. Neither coefficient changes the raw result score.
    """
    for name, value in (
        ("current", current),
        ("result", result),
        ("eta", eta),
        ("assistance_coefficient", assistance_coefficient),
        ("difficulty_coefficient", difficulty_coefficient),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    effective_eta = eta * assistance_coefficient * difficulty_coefficient
    return current + effective_eta * (result - current)


def update_level_from_attempt(current: float, attempt: Attempt, eta: float) -> float:
    """Apply the same level-update rule to any recorded attempt category."""
    if not isinstance(attempt, Attempt):
        raise ValueError("attempt must be an Attempt")
    return update_level(
        current,
        attempt.score,
        eta,
        assistance_coefficient=attempt.assistance_coefficient,
        difficulty_coefficient=attempt.difficulty_coefficient,
    )


def weighted_gap(
    current: Mapping[str, float],
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the sum of weighted positive shortfalls from target levels.

    Current and target maps must describe the same concepts. If weights are
    omitted, every concept has equal weight. The result is non-negative and
    may exceed 1 because the shortfalls are summed rather than averaged.
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

    # Weights may be any finite, non-negative numbers.
    for concept, weight in weights.items():
        if weight < 0 or not isfinite(weight):
            raise ValueError(f"weights[{concept!r}] must be finite and non-negative")

    # A zero weight sum contributes no gap.
    if sum(weights.values()) == 0:
        return 0.0

    # Sum the weighted positive shortfalls exactly as defined in the spec.
    gap = sum(
        weights[concept] * max(target[concept] - current[concept], 0.0)
        for concept in concepts
    )
    if not isfinite(gap):
        raise ValueError("weighted gap must be finite")
    return gap


def progression(initial_gap: float, current_gap: float) -> float:
    """Return the clamped fraction of an arbitrary non-negative initial gap closed.

    Gaps need not be in [0, 1]: a weighted sum of concept shortfalls can
    exceed 1. An initial gap of zero is treated as complete (1.0) if the
    current gap is also zero; if a gap appears later, progression is 0.0.
    """
    # Weighted sums are not normalized, but gaps must remain finite and >= 0.
    for name, value in (("initial_gap", initial_gap), ("current_gap", current_gap)):
        if value < 0 or not isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")

    # Avoid dividing by zero: no initial gap means nothing was left to close.
    if initial_gap == 0:
        return 1.0 if current_gap == 0 else 0.0

    # Fraction closed = (initial - current) / initial. Clamp it so a growing
    # gap cannot produce negative progress and overshoot cannot exceed 100%.
    return min(1.0, max(0.0, (initial_gap - current_gap) / initial_gap))
