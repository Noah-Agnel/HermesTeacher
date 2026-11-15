"""Concept graph proposal, validation and correction.

The graph is proposed from structured input: the agent extracts concepts and
prerequisite relations from the learner's documents and passes them here. This
module does not invent that extraction. It validates the proposal, detects
cycles and unknown prerequisites, checks that evidence points at real passages,
and lets the learner correct the graph through explicit operations.

A proposal is immutable: every correction returns a new proposal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from .domain import Concept, topological_order


def _check_keys(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)!r}")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class Evidence:
    """A citation of a document that supports a concept or a relation.

    The backend does not split documents into passages, so a citation points
    at a document and optionally at a page and a short quote.
    """

    document_id: str
    page: int | None = None
    quote: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.document_id, "document_id")
        if self.page is not None:
            if (
                not isinstance(self.page, int)
                or isinstance(self.page, bool)
                or self.page < 1
            ):
                raise ValueError("page must be a positive integer or None")
        if self.quote is not None and not isinstance(self.quote, str):
            raise ValueError("quote must be a string or None")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page": self.page,
            "quote": self.quote,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "Evidence":
        if not isinstance(data, Mapping):
            raise ValueError("evidence must be a mapping")
        _check_keys(data, {"document_id", "page", "quote"}, "evidence")
        return cls(
            document_id=_require_string(data.get("document_id"), "document_id"),
            page=data.get("page"),
            quote=data.get("quote"),
        )


@dataclass(frozen=True)
class ProposedConcept:
    """A concept proposed for the graph, with optional evidence."""

    concept_id: str
    label: str
    description: str = ""
    target_level: float = 1.0
    weight: float = 1.0
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.concept_id, "concept_id")
        _require_string(self.label, "label")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if not isinstance(self.target_level, (int, float)) or not 0 <= self.target_level <= 1:
            raise ValueError("target_level must be between 0 and 1")
        if not isinstance(self.weight, (int, float)) or self.weight < 0 or not isfinite(self.weight):
            raise ValueError("weight must be finite and non-negative")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise ValueError("evidence must contain Evidence items")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "description": self.description,
            "target_level": self.target_level,
            "weight": self.weight,
            "evidence": [item.to_mapping() for item in self.evidence],
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "ProposedConcept":
        if not isinstance(data, Mapping):
            raise ValueError("concept must be a mapping")
        _check_keys(
            data,
            {"concept_id", "label", "description", "target_level", "weight", "evidence"},
            "concept",
        )
        evidence_data = data.get("evidence", [])
        if not isinstance(evidence_data, list):
            raise ValueError("evidence must be a list")
        return cls(
            concept_id=_require_string(data.get("concept_id"), "concept_id"),
            label=_require_string(data.get("label"), "label"),
            description=data.get("description", ""),
            target_level=data.get("target_level", 1.0),
            weight=data.get("weight", 1.0),
            evidence=tuple(Evidence.from_mapping(item) for item in evidence_data),
        )


@dataclass(frozen=True)
class ProposedEdge:
    """A proposed prerequisite relation: ``prerequisite_id`` before ``concept_id``."""

    prerequisite_id: str
    concept_id: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.prerequisite_id, "prerequisite_id")
        _require_string(self.concept_id, "concept_id")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise ValueError("evidence must contain Evidence items")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prerequisite_id": self.prerequisite_id,
            "concept_id": self.concept_id,
            "evidence": [item.to_mapping() for item in self.evidence],
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "ProposedEdge":
        if not isinstance(data, Mapping):
            raise ValueError("edge must be a mapping")
        _check_keys(data, {"prerequisite_id", "concept_id", "evidence"}, "edge")
        evidence_data = data.get("evidence", [])
        if not isinstance(evidence_data, list):
            raise ValueError("evidence must be a list")
        return cls(
            prerequisite_id=_require_string(data.get("prerequisite_id"), "prerequisite_id"),
            concept_id=_require_string(data.get("concept_id"), "concept_id"),
            evidence=tuple(Evidence.from_mapping(item) for item in evidence_data),
        )


@dataclass(frozen=True)
class GraphIssue:
    """A problem found in a proposal, with a stable code."""

    code: str
    message: str
    concept_id: str | None = None
    prerequisite_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """The issues found in a proposal; ``is_valid`` means it can be used."""

    issues: tuple[GraphIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ConceptGraphProposal:
    """An immutable proposed concept graph that the learner can correct."""

    graph_id: str
    concepts: tuple[ProposedConcept, ...] = ()
    edges: tuple[ProposedEdge, ...] = ()
    source_document_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        _require_string(self.graph_id, "graph_id")
        object.__setattr__(self, "concepts", tuple(self.concepts))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "source_document_ids", tuple(self.source_document_ids))
        for concept in self.concepts:
            if not isinstance(concept, ProposedConcept):
                raise ValueError("concepts must contain ProposedConcept items")
        for edge in self.edges:
            if not isinstance(edge, ProposedEdge):
                raise ValueError("edges must contain ProposedEdge items")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")

    @staticmethod
    def _normalize_known_documents(
        known_documents: Iterable[str] | Mapping[str, Any],
    ) -> set[str]:
        if isinstance(known_documents, Mapping):
            return set(known_documents.keys())
        return set(known_documents)

    def validate(
        self,
        known_documents: Iterable[str] | Mapping[str, Any] | None = None,
    ) -> ValidationReport:
        """Return the issues in the proposal.

        ``known_documents`` is an optional collection of document ids (or a
        mapping keyed by them). When provided, evidence citing an unknown
        document is reported.
        """
        issues: list[GraphIssue] = []
        concept_ids = [concept.concept_id for concept in self.concepts]
        known_concepts = set(concept_ids)

        seen: set[str] = set()
        for concept_id in concept_ids:
            if concept_id in seen:
                issues.append(
                    GraphIssue(
                        "duplicate_concept",
                        f"concept {concept_id!r} appears more than once",
                        concept_id=concept_id,
                    )
                )
            seen.add(concept_id)

        if not concept_ids:
            issues.append(GraphIssue("empty_graph", "the proposal has no concepts"))

        edge_keys: set[tuple[str, str]] = set()
        for edge in self.edges:
            key = (edge.prerequisite_id, edge.concept_id)
            if key in edge_keys:
                issues.append(
                    GraphIssue(
                        "duplicate_edge",
                        f"edge {edge.prerequisite_id!r} -> {edge.concept_id!r} appears more than once",
                        concept_id=edge.concept_id,
                        prerequisite_id=edge.prerequisite_id,
                    )
                )
            edge_keys.add(key)
            if edge.prerequisite_id not in known_concepts:
                issues.append(
                    GraphIssue(
                        "unknown_prerequisite",
                        f"prerequisite {edge.prerequisite_id!r} is not a proposed concept",
                        concept_id=edge.concept_id,
                        prerequisite_id=edge.prerequisite_id,
                    )
                )
            if edge.concept_id not in known_concepts:
                issues.append(
                    GraphIssue(
                        "unknown_concept",
                        f"edge targets unknown concept {edge.concept_id!r}",
                        concept_id=edge.concept_id,
                        prerequisite_id=edge.prerequisite_id,
                    )
                )
            if edge.prerequisite_id == edge.concept_id:
                issues.append(
                    GraphIssue(
                        "self_loop",
                        f"concept {edge.concept_id!r} cannot be its own prerequisite",
                        concept_id=edge.concept_id,
                    )
                )

        if known_documents is not None:
            known = self._normalize_known_documents(known_documents)
            for concept in self.concepts:
                for item in concept.evidence:
                    if item.document_id not in known:
                        issues.append(
                            GraphIssue(
                                "unknown_document",
                                f"concept {concept.concept_id!r} cites unknown document {item.document_id!r}",
                                concept_id=concept.concept_id,
                            )
                        )
            for edge in self.edges:
                for item in edge.evidence:
                    if item.document_id not in known:
                        issues.append(
                            GraphIssue(
                                "unknown_document",
                                f"edge {edge.prerequisite_id!r} -> {edge.concept_id!r} cites unknown document {item.document_id!r}",
                                concept_id=edge.concept_id,
                                prerequisite_id=edge.prerequisite_id,
                            )
                        )

        # Cycle detection on the structurally known edges.
        prerequisites = {concept_id: set() for concept_id in concept_ids}
        for edge in self.edges:
            if edge.prerequisite_id in prerequisites and edge.concept_id in prerequisites:
                prerequisites[edge.concept_id].add(edge.prerequisite_id)
        domain_concepts = {
            concept.concept_id: Concept(
                concept.concept_id,
                prerequisites=frozenset(prerequisites[concept.concept_id]),
                target_level=concept.target_level,
                weight=concept.weight,
            )
            for concept in self.concepts
        }
        try:
            topological_order(domain_concepts)
        except ValueError as error:
            issues.append(GraphIssue("cycle", str(error)))

        return ValidationReport(issues=tuple(issues))

    def to_domain_concepts(self) -> dict[str, Concept]:
        """Convert a valid proposal into domain concepts; raise otherwise."""
        report = self.validate()
        if not report.is_valid:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
            raise ValueError(f"proposal is not valid: {details}")
        prerequisites = {concept.concept_id: set() for concept in self.concepts}
        for edge in self.edges:
            prerequisites[edge.concept_id].add(edge.prerequisite_id)
        return {
            concept.concept_id: Concept(
                concept.concept_id,
                prerequisites=frozenset(prerequisites[concept.concept_id]),
                target_level=concept.target_level,
                weight=concept.weight,
            )
            for concept in self.concepts
        }

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic prerequisite-safe order for a valid proposal."""
        return topological_order(self.to_domain_concepts())

    def adjacency_list(self) -> dict[str, list[str]]:
        """Return the prerequisite adjacency list: concept -> prerequisites.

        Each key is a proposed concept; its value lists the concept ids that
        must be mastered first. The graph is directed and acyclic.
        """
        adjacency: dict[str, list[str]] = {
            concept.concept_id: [] for concept in self.concepts
        }
        for edge in self.edges:
            if edge.concept_id in adjacency and edge.prerequisite_id in adjacency:
                adjacency[edge.concept_id].append(edge.prerequisite_id)
        for concept_id in adjacency:
            adjacency[concept_id] = sorted(set(adjacency[concept_id]))
        return adjacency

    def add_concept(self, concept: ProposedConcept) -> "ConceptGraphProposal":
        if not isinstance(concept, ProposedConcept):
            raise ValueError("concept must be a ProposedConcept")
        if any(existing.concept_id == concept.concept_id for existing in self.concepts):
            raise ValueError(f"concept {concept.concept_id!r} already exists")
        return replace(self, concepts=self.concepts + (concept,))

    def remove_concept(self, concept_id: str) -> "ConceptGraphProposal":
        if not any(concept.concept_id == concept_id for concept in self.concepts):
            raise ValueError(f"concept {concept_id!r} does not exist")
        concepts = tuple(
            concept for concept in self.concepts if concept.concept_id != concept_id
        )
        edges = tuple(
            edge
            for edge in self.edges
            if edge.concept_id != concept_id and edge.prerequisite_id != concept_id
        )
        return replace(self, concepts=concepts, edges=edges)

    def add_edge(self, edge: ProposedEdge) -> "ConceptGraphProposal":
        if not isinstance(edge, ProposedEdge):
            raise ValueError("edge must be a ProposedEdge")
        known = {concept.concept_id for concept in self.concepts}
        if edge.prerequisite_id not in known or edge.concept_id not in known:
            raise ValueError("edge endpoints must reference existing concepts")
        if any(
            (existing.prerequisite_id, existing.concept_id)
            == (edge.prerequisite_id, edge.concept_id)
            for existing in self.edges
        ):
            raise ValueError("edge already exists")
        return replace(self, edges=self.edges + (edge,))

    def remove_edge(self, prerequisite_id: str, concept_id: str) -> "ConceptGraphProposal":
        edges = tuple(
            edge
            for edge in self.edges
            if not (
                edge.prerequisite_id == prerequisite_id and edge.concept_id == concept_id
            )
        )
        if len(edges) == len(self.edges):
            raise ValueError("edge does not exist")
        return replace(self, edges=edges)

    def rename_concept(self, concept_id: str, label: str) -> "ConceptGraphProposal":
        return self._map_concept(concept_id, lambda concept: replace(concept, label=label))

    def set_target_level(self, concept_id: str, value: float) -> "ConceptGraphProposal":
        return self._map_concept(concept_id, lambda concept: replace(concept, target_level=value))

    def set_weight(self, concept_id: str, value: float) -> "ConceptGraphProposal":
        return self._map_concept(concept_id, lambda concept: replace(concept, weight=value))

    def _map_concept(self, concept_id: str, transform) -> "ConceptGraphProposal":
        if not any(concept.concept_id == concept_id for concept in self.concepts):
            raise ValueError(f"concept {concept_id!r} does not exist")
        concepts = tuple(
            transform(concept) if concept.concept_id == concept_id else concept
            for concept in self.concepts
        )
        return replace(self, concepts=concepts)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "source_document_ids": list(self.source_document_ids),
            "notes": self.notes,
            "concepts": [concept.to_mapping() for concept in self.concepts],
            "edges": [edge.to_mapping() for edge in self.edges],
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "ConceptGraphProposal":
        if not isinstance(data, Mapping):
            raise ValueError("proposal must be a mapping")
        _check_keys(
            data,
            {"graph_id", "concepts", "edges", "source_document_ids", "notes"},
            "proposal",
        )
        concepts_data = data.get("concepts", [])
        edges_data = data.get("edges", [])
        if not isinstance(concepts_data, list) or not isinstance(edges_data, list):
            raise ValueError("concepts and edges must be lists")
        source_document_ids = data.get("source_document_ids", [])
        if not isinstance(source_document_ids, list) or any(
            not isinstance(item, str) for item in source_document_ids
        ):
            raise ValueError("source_document_ids must be a list of strings")
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("notes must be a string")
        return cls(
            graph_id=_require_string(data.get("graph_id"), "graph_id"),
            concepts=tuple(ProposedConcept.from_mapping(item) for item in concepts_data),
            edges=tuple(ProposedEdge.from_mapping(item) for item in edges_data),
            source_document_ids=tuple(source_document_ids),
            notes=notes,
        )
