import unittest

from policy_utils import (
    cache_is_fresh,
    can_transition,
    effective_timeout,
    exponential_backoff,
    inventory_after_reservation,
    quota_percent,
    resolve_setting,
    retryable_status,
    safe_divide,
)


class PolicyUtilsTest(unittest.TestCase):
    def test_transition_uses_explicit_state_graph(self):
        self.assertTrue(can_transition("queued", "running"))
        self.assertFalse(can_transition("succeeded", "running"))

    def test_backoff_is_exponential_and_capped(self):
        self.assertEqual(exponential_backoff(3, 2, 10), 8)
        self.assertEqual(exponential_backoff(5, 2, 10), 10)

    def test_quota_percent_returns_percentage(self):
        self.assertEqual(quota_percent(25, 100), 25.0)

    def test_inventory_accepts_exact_remaining_amount(self):
        self.assertEqual(inventory_after_reservation(5, 5), 0)
        self.assertIsNone(inventory_after_reservation(5, 6))

    def test_cache_expires_at_ttl_boundary(self):
        self.assertFalse(cache_is_fresh(100.0, 10.0, 110.0))
        self.assertTrue(cache_is_fresh(100.0, 10.0, 109.9))

    def test_effective_timeout_never_exceeds_maximum(self):
        self.assertEqual(effective_timeout(30, 10), 10)
        self.assertEqual(effective_timeout(5, 10), 5)

    def test_retryable_status_only_transient_failures(self):
        self.assertTrue(retryable_status(429))
        self.assertTrue(retryable_status(503))
        self.assertFalse(retryable_status(400))

    def test_setting_precedence_environment_then_file_then_default(self):
        self.assertEqual(resolve_setting({"VALUE": "env"}, "file", "default"), "env")
        self.assertEqual(resolve_setting({}, "file", "default"), "file")

    def test_safe_divide_returns_none_for_zero_denominator(self):
        self.assertIsNone(safe_divide(5, 0))
        self.assertEqual(safe_divide(6, 3), 2)


if __name__ == "__main__":
    unittest.main()
