import unittest

from backend.tutor.domain import update_level, weighted_gap, progression


class UpdateLevelTests(unittest.TestCase):
    def test_updates_level_with_weighted_result(self):
        self.assertAlmostEqual(update_level(0.4, 0.8, 0.5), 0.6)

    def test_aided_attempt_does_not_change_level(self):
        self.assertEqual(update_level(0.4, 1.0, 0.5, aided=True), 0.4)

    def test_rejects_values_outside_unit_interval(self):
        for args in ((-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, -0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                update_level(*args)


class WeightedGapTests(unittest.TestCase):
    def test_weighted_gap_ignores_concepts_at_or_above_target(self):
        current = {"a": 0.2, "b": 0.8, "c": 0.9}
        target = {"a": 0.8, "b": 0.5, "c": 0.9}
        weights = {"a": 2.0, "b": 1.0, "c": 1.0}
        self.assertAlmostEqual(weighted_gap(current, target, weights), 0.3)

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

    def test_rejects_gaps_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            progression(1.1, 0.1)


if __name__ == "__main__":
    unittest.main()
