import unittest

from quota import can_reserve


class QuotaTest(unittest.TestCase):
    def test_accepts_exact_remaining_quota(self):
        self.assertTrue(can_reserve(100, 100))

    def test_rejects_overuse(self):
        self.assertFalse(can_reserve(100, 101))


if __name__ == "__main__":
    unittest.main()
