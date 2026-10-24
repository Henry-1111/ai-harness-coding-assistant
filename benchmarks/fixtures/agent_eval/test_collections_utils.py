import unittest

from collections_utils import (
    chunk,
    clamp,
    deduplicate,
    flatten,
    intersection,
    moving_average,
    paginate,
    rotate_left,
    top_n,
)


class CollectionsUtilsTest(unittest.TestCase):
    def test_deduplicate_preserves_order(self):
        self.assertEqual(deduplicate([3, 1, 3, 2, 1]), [3, 1, 2])

    def test_chunk_has_no_empty_tail(self):
        self.assertEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_flatten_keeps_every_item(self):
        self.assertEqual(flatten([[1, 2], [3], [4, 5]]), [1, 2, 3, 4, 5])

    def test_rotate_wraps_large_counts(self):
        self.assertEqual(rotate_left([1, 2, 3], 4), [2, 3, 1])

    def test_clamp_respects_bounds(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(11, 0, 10), 10)

    def test_moving_average_only_complete_windows(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_paginate_uses_one_based_page_numbers(self):
        self.assertEqual(paginate(list(range(10)), 2, 3), [3, 4, 5])

    def test_intersection_preserves_left_order(self):
        self.assertEqual(intersection([3, 1, 2, 1], [1, 3]), [3, 1])

    def test_top_n_returns_largest_values_descending(self):
        self.assertEqual(top_n([4, 1, 7, 3], 2), [7, 4])


if __name__ == "__main__":
    unittest.main()
