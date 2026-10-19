import unittest

from src.simulation.rules import StopApproach


class StopApproachTests(unittest.TestCase):
    def test_stop_before_sign_allows_passage(self):
        rule = StopApproach()
        self.assertIsNone(rule.update(8.0, 3.0))
        self.assertIsNone(rule.update(2.0, 0.0))
        self.assertFalse(rule.update(-0.5, 1.0))
        self.assertIsNone(rule.update(-2.0, 2.0))

    def test_rolling_through_is_one_violation(self):
        rule = StopApproach()
        rule.update(3.0, 1.0)
        self.assertTrue(rule.update(-0.5, 1.0))
        self.assertIsNone(rule.update(-2.0, 1.0))

    def test_stop_after_sign_does_not_clear_violation(self):
        rule = StopApproach()
        rule.update(2.0, 1.0)
        self.assertTrue(rule.update(-1.0, 1.0))
        self.assertIsNone(rule.update(-2.0, 0.0))

    def test_new_approach_requires_new_stop(self):
        rule = StopApproach()
        rule.update(2.0, 0.0)
        self.assertFalse(rule.update(-1.0, 1.0))
        rule.update(8.0, 1.0)
        self.assertTrue(rule.update(-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
