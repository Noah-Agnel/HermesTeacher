import unittest

try:
    from backend.tutor import graph
except ImportError:
    graph = None


def make_proposal(*, graph_id="graph-1", concepts=(), edges=(), **kwargs):
    proposals = getattr(graph, "ConceptGraphProposal", None)
    if proposals is None:
        return None
    return proposals(
        graph_id=graph_id,
        concepts=tuple(concepts),
        edges=tuple(edges),
        **kwargs,
    )


class ProposedConceptTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(graph, "graph module should be implemented")

    def test_rejects_empty_ids_and_labels(self):
        concept_type = graph.ProposedConcept
        with self.assertRaises(ValueError):
            concept_type("", "Label")
        with self.assertRaises(ValueError):
            concept_type("a", "")

    def test_rejects_invalid_levels_and_weights(self):
        concept_type = graph.ProposedConcept
        with self.assertRaises(ValueError):
            concept_type("a", "Label", target_level=1.5)
        with self.assertRaises(ValueError):
            concept_type("a", "Label", target_level=-0.1)
        with self.assertRaises(ValueError):
            concept_type("a", "Label", weight=-1.0)


class ProposalValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(graph, "graph module should be implemented")

    def test_valid_proposal_has_no_issues(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[graph.ProposedEdge("a", "b")],
        )

        report = proposal.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.issues, ())

    def test_unknown_prerequisite_is_reported(self):
        proposal = make_proposal(
            concepts=[graph.ProposedConcept("b", "B")],
            edges=[graph.ProposedEdge("a", "b")],
        )

        report = proposal.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("unknown_prerequisite", [issue.code for issue in report.issues])

    def test_self_loop_is_reported(self):
        proposal = make_proposal(
            concepts=[graph.ProposedConcept("a", "A")],
            edges=[graph.ProposedEdge("a", "a")],
        )

        report = proposal.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("self_loop", [issue.code for issue in report.issues])

    def test_cycle_is_reported(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[
                graph.ProposedEdge("a", "b"),
                graph.ProposedEdge("b", "a"),
            ],
        )

        report = proposal.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("cycle", [issue.code for issue in report.issues])

    def test_duplicate_concepts_are_reported(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("a", "A again"),
            ]
        )

        report = proposal.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_concept", [issue.code for issue in report.issues])

    def test_duplicate_edges_are_reported(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[
                graph.ProposedEdge("a", "b"),
                graph.ProposedEdge("a", "b"),
            ],
        )

        report = proposal.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_edge", [issue.code for issue in report.issues])

    def test_unknown_document_is_reported_when_known_documents_given(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept(
                    "a",
                    "A",
                    evidence=(graph.Evidence("doc-1", page=1),),
                )
            ]
        )

        report = proposal.validate(known_documents={"doc-2"})

        self.assertFalse(report.is_valid)
        self.assertIn("unknown_document", [issue.code for issue in report.issues])

    def test_known_document_produces_no_issue(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept(
                    "a",
                    "A",
                    evidence=(graph.Evidence("doc-1", page=1),),
                )
            ]
        )

        report = proposal.validate(known_documents={"doc-1"})

        self.assertTrue(report.is_valid)


class DomainConversionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(graph, "graph module should be implemented")

    def test_to_domain_concepts_builds_prerequisites(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A", target_level=0.8, weight=2.0),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[graph.ProposedEdge("a", "b")],
        )

        concepts = proposal.to_domain_concepts()

        self.assertIn("a", concepts["b"].prerequisites)
        self.assertEqual(concepts["a"].target_level, 0.8)
        self.assertEqual(concepts["a"].weight, 2.0)

    def test_to_domain_concepts_raises_on_invalid_proposal(self):
        proposal = make_proposal(
            concepts=[graph.ProposedConcept("b", "B")],
            edges=[graph.ProposedEdge("a", "b")],
        )

        with self.assertRaises(ValueError):
            proposal.to_domain_concepts()

    def test_topological_order_respects_prerequisites(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
                graph.ProposedConcept("c", "C"),
            ],
            edges=[
                graph.ProposedEdge("a", "b"),
                graph.ProposedEdge("b", "c"),
            ],
        )

        order = proposal.topological_order()

        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("b"), order.index("c"))


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(graph, "graph module should be implemented")

    def test_add_concept(self):
        proposal = make_proposal()
        updated = proposal.add_concept(graph.ProposedConcept("a", "A"))

        self.assertIn("a", [concept.concept_id for concept in updated.concepts])

    def test_add_concept_rejects_duplicate(self):
        proposal = make_proposal(concepts=[graph.ProposedConcept("a", "A")])

        with self.assertRaises(ValueError):
            proposal.add_concept(graph.ProposedConcept("a", "A again"))

    def test_remove_concept_removes_its_edges(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[graph.ProposedEdge("a", "b")],
        )

        updated = proposal.remove_concept("a")

        self.assertEqual([c.concept_id for c in updated.concepts], ["b"])
        self.assertEqual(updated.edges, ())

    def test_add_and_remove_edge(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept("a", "A"),
                graph.ProposedConcept("b", "B"),
            ]
        )

        updated = proposal.add_edge(graph.ProposedEdge("a", "b"))
        self.assertIn(("a", "b"), [(e.prerequisite_id, e.concept_id) for e in updated.edges])

        removed = updated.remove_edge("a", "b")
        self.assertEqual(removed.edges, ())

    def test_add_edge_rejects_unknown_concept(self):
        proposal = make_proposal(concepts=[graph.ProposedConcept("a", "A")])

        with self.assertRaises(ValueError):
            proposal.add_edge(graph.ProposedEdge("a", "missing"))

    def test_rename_concept(self):
        proposal = make_proposal(concepts=[graph.ProposedConcept("a", "A")])

        updated = proposal.rename_concept("a", "Nouveau nom")

        self.assertEqual(updated.concepts[0].label, "Nouveau nom")

    def test_set_target_level_and_weight(self):
        proposal = make_proposal(concepts=[graph.ProposedConcept("a", "A")])

        updated = proposal.set_target_level("a", 0.9).set_weight("a", 3.0)

        self.assertEqual(updated.concepts[0].target_level, 0.9)
        self.assertEqual(updated.concepts[0].weight, 3.0)

    def test_corrections_do_not_mutate_the_original(self):
        proposal = make_proposal(concepts=[graph.ProposedConcept("a", "A")])

        proposal.add_concept(graph.ProposedConcept("b", "B"))

        self.assertEqual([c.concept_id for c in proposal.concepts], ["a"])


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(graph, "graph module should be implemented")

    def test_mapping_round_trip(self):
        proposal = make_proposal(
            concepts=[
                graph.ProposedConcept(
                    "a",
                    "A",
                    description="Description",
                    target_level=0.8,
                    weight=2.0,
                    evidence=(graph.Evidence("doc-1", page=1, quote="quote"),),
                ),
                graph.ProposedConcept("b", "B"),
            ],
            edges=[graph.ProposedEdge("a", "b", evidence=(graph.Evidence("doc-1", page=1),))],
            source_document_ids=["doc-1"],
            notes="proposé",
        )

        data = proposal.to_mapping()
        restored = graph.ConceptGraphProposal.from_mapping(data)

        self.assertEqual(restored.to_mapping(), data)

    def test_from_mapping_rejects_malformed_data(self):
        with self.assertRaises(ValueError):
            graph.ConceptGraphProposal.from_mapping({"concepts": []})
        with self.assertRaises(ValueError):
            graph.ConceptGraphProposal.from_mapping(
                {"graph_id": "graph-1", "concepts": "not-a-list"}
            )

    def test_from_mapping_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            graph.ConceptGraphProposal.from_mapping(
                {
                    "graph_id": "graph-1",
                    "concepts": [{"concept_id": "a", "label": "A"}],
                    "edges": [{"prerequisite_id": "a", "concept_id": "b"}],
                    "unexpected": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
