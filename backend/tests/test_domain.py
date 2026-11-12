import unittest

from backend.tutor import domain
from backend.tutor.domain import update_level, weighted_gap, progression


class UpdateLevelTests(unittest.TestCase):
    def test_updates_level_with_weighted_result(self):
        self.assertAlmostEqual(update_level(0.4, 0.8, 0.5), 0.6)

    def test_help_coefficient_scales_effective_learning_rate(self):
        self.assertAlmostEqual(update_level(0.4, 0.8, 0.5, 0.25), 0.45)

    def test_zero_help_coefficient_does_not_change_level(self):
        self.assertEqual(update_level(0.4, 1.0, 0.5, 0.0), 0.4)

    def test_rejects_invalid_help_coefficients(self):
        for coefficient in (-0.1, 1.1, float("nan")):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                update_level(0.4, 0.8, 0.5, coefficient)

    def test_rejects_values_outside_unit_interval(self):
        for args in ((-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, -0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                update_level(*args)


class AttemptTypeTests(unittest.TestCase):
    def test_difficulty_coefficient_weights_evidence_without_changing_raw_score(self):
        attempt_type = getattr(domain, "AttemptType", None)
        self.assertIsNotNone(attempt_type, "AttemptType should be defined")
        attempt_class = getattr(domain, "Attempt", None)
        self.assertIsNotNone(attempt_class, "Attempt should be defined")
        self.assertIn("difficulty_coefficient", attempt_class.__dataclass_fields__)

        attempt = attempt_class(
            kind=attempt_type.EXERCISE,
            score=0.8,
            assistance_coefficient=0.5,
            difficulty_coefficient=0.25,
        )
        self.assertEqual(attempt.score, 0.8)
        self.assertAlmostEqual(domain.update_level_from_attempt(0.4, attempt, eta=0.5), 0.425)

    def test_attempt_categories_share_update_rule_for_now(self):
        attempt_type = getattr(domain, "AttemptType", None)
        self.assertIsNotNone(attempt_type, "AttemptType should be defined")
        attempt = getattr(domain, "Attempt", None)
        self.assertIsNotNone(attempt, "Attempt should be defined")
        apply_attempt = getattr(domain, "update_level_from_attempt", None)
        self.assertIsNotNone(apply_attempt, "update_level_from_attempt should be defined")

        for kind in (attempt_type.EXERCISE, attempt_type.EVALUATION, attempt_type.FINAL_EXAM):
            with self.subTest(kind=kind):
                item = attempt(kind=kind, score=0.8, assistance_coefficient=0.25)
                self.assertAlmostEqual(apply_attempt(0.4, item, eta=0.5), 0.45)


class ConceptPlanningTests(unittest.TestCase):
    def test_topological_order_puts_prerequisites_first(self):
        concept_type = getattr(domain, "Concept", None)
        self.assertIsNotNone(concept_type, "Concept should be defined")
        order_graph = getattr(domain, "topological_order", None)
        self.assertIsNotNone(order_graph, "topological_order should be defined")

        concepts = {
            "advanced": concept_type("advanced", prerequisites={"base"}),
            "base": concept_type("base"),
        }
        order = order_graph(concepts)
        self.assertLess(order.index("base"), order.index("advanced"))

    def test_topological_order_rejects_missing_prerequisites_and_cycles(self):
        concept_type = getattr(domain, "Concept", None)
        self.assertIsNotNone(concept_type, "Concept should be defined")
        order_graph = getattr(domain, "topological_order", None)
        self.assertIsNotNone(order_graph, "topological_order should be defined")

        with self.assertRaises(ValueError):
            order_graph({"a": concept_type("a", prerequisites={"missing"})})
        cyclic = {
            "a": concept_type("a", prerequisites={"b"}),
            "b": concept_type("b", prerequisites={"a"}),
        }
        with self.assertRaises(ValueError):
            order_graph(cyclic)

    def test_plan_includes_under_mastered_prerequisites_and_prioritizes_weight(self):
        concept_type = getattr(domain, "Concept", None)
        self.assertIsNotNone(concept_type, "Concept should be defined")
        plan = getattr(domain, "plan_concepts", None)
        self.assertIsNotNone(plan, "plan_concepts should be defined")

        concepts = {
            "foundation": concept_type("foundation", target_level=0.4, weight=1.0),
            "other": concept_type("other", target_level=0.9, weight=2.0),
            "goal": concept_type(
                "goal", prerequisites={"foundation"}, target_level=1.0, weight=3.0
            ),
        }
        current = {"foundation": 0.4, "other": 0.2, "goal": 0.3}
        order = plan(concepts, current, prerequisite_threshold=0.6)

        self.assertEqual(set(order), {"foundation", "other", "goal"})
        self.assertLess(order.index("foundation"), order.index("goal"))
        self.assertEqual(order[0], "other")

    def test_default_threshold_is_seventy_five_percent(self):
        threshold = getattr(domain, "DEFAULT_PREREQUISITE_THRESHOLD", None)
        self.assertIsNotNone(threshold, "a default prerequisite threshold should be defined")
        self.assertEqual(threshold, 0.75)

        plan = domain.plan_concepts
        concepts = {
            "foundation": domain.Concept("foundation", target_level=0.6),
            "goal": domain.Concept("goal", prerequisites={"foundation"}, target_level=0.9),
        }
        current = {"foundation": 0.7, "goal": 0.3}
        order = plan(concepts, current)
        self.assertEqual(order[0], "foundation")

    def test_plan_rejects_invalid_levels_and_threshold(self):
        concept_type = getattr(domain, "Concept", None)
        self.assertIsNotNone(concept_type, "Concept should be defined")
        plan = getattr(domain, "plan_concepts", None)
        self.assertIsNotNone(plan, "plan_concepts should be defined")
        concepts = {"a": concept_type("a")}

        with self.assertRaises(ValueError):
            plan(concepts, {"a": 1.1}, prerequisite_threshold=0.6)
        with self.assertRaises(ValueError):
            plan(concepts, {"a": 0.0}, prerequisite_threshold=1.1)


class WeightedGapTests(unittest.TestCase):
    def test_weighted_gap_is_sum_of_weighted_shortfalls(self):
        current = {"a": 0.2, "b": 0.8, "c": 0.9}
        target = {"a": 0.8, "b": 0.5, "c": 0.9}
        weights = {"a": 2.0, "b": 1.0, "c": 1.0}
        self.assertAlmostEqual(weighted_gap(current, target, weights), 1.2)

    def test_rejects_invalid_levels_and_concept_sets(self):
        with self.assertRaises(ValueError):
            weighted_gap({"a": 1.2}, {"a": 1.0})
        with self.assertRaises(ValueError):
            weighted_gap({"a": 0.2}, {"b": 0.8})


class ProgressionTests(unittest.TestCase):
    def test_progression_is_fraction_of_initial_gap_closed(self):
        self.assertAlmostEqual(progression(0.8, 0.2), 0.75)

    def test_progression_is_clamped_between_zero_and_one(self):
        self.assertEqual(progression(0.5, 0.8), 0.0)
        self.assertEqual(progression(0.5, 0.0), 1.0)

    def test_zero_initial_gap_does_not_divide_by_zero(self):
        self.assertEqual(progression(0.0, 0.0), 1.0)
        self.assertEqual(progression(0.0, 0.2), 0.0)

    def test_progression_accepts_unscaled_weighted_gaps(self):
        self.assertAlmostEqual(progression(4.0, 1.0), 0.75)

    def test_rejects_negative_gaps(self):
        with self.assertRaises(ValueError):
            progression(-0.1, 0.0)

    def test_rejects_non_finite_gaps(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                progression(value, 0.0)


if __name__ == "__main__":
    unittest.main()
