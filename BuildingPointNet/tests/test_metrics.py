import unittest

import numpy as np

from building_pointnet.dataset import window_permutation
from building_pointnet.metrics import average_precision, threshold_metrics


class TestAveragePrecision(unittest.TestCase):
    def test_constant_scores_are_order_invariant(self):
        # 8 positives, 2 negatives, all scores tied -> AP = 0.8.
        labels = np.array([1] * 8 + [0] * 2, dtype=np.int8)
        probs = np.full(10, 0.1, dtype=np.float64)
        self.assertAlmostEqual(average_precision(labels, probs), 0.8, places=9)

        # Same multiset, reversed row order.
        self.assertAlmostEqual(
            average_precision(labels[::-1], probs[::-1]), 0.8, places=9
        )

    def test_random_scores_are_row_permutation_invariant(self):
        rng = np.random.RandomState(7)
        labels = rng.randint(0, 2, size=5000).astype(np.int8)
        probs = rng.rand(5000)
        perm = rng.permutation(5000)
        self.assertAlmostEqual(
            average_precision(labels, probs),
            average_precision(labels[perm], probs[perm]),
            places=12,
        )

    def test_negative_labels_are_ignored(self):
        labels = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)
        probs = np.array([0.9, 0.1, 0.99, 0.6, 0.4, 0.0])
        labels2 = labels[labels >= 0]
        probs2 = probs[labels >= 0]
        self.assertAlmostEqual(
            average_precision(labels, probs),
            average_precision(labels2, probs2),
            places=12,
        )


class TestWindowPermutation(unittest.TestCase):
    def test_deterministic_and_center_dependent(self):
        p1 = window_permutation(1000, 405450.0, 3112700.0, 20261028)
        p2 = window_permutation(1000, 405450.0, 3112700.0, 20261028)
        p3 = window_permutation(1000, 405465.0, 3112700.0, 20261028)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertFalse(np.array_equal(p1, p3))
        # A permutation, not a resampling: same multiset of indices.
        self.assertTrue(np.array_equal(np.sort(p1), np.arange(1000)))


class TestThresholdMetrics(unittest.TestCase):
    def test_constant_threshold_metrics(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        m = threshold_metrics(labels, probs, 0.5)
        self.assertAlmostEqual(m["building_f1"], 1.0)
        self.assertAlmostEqual(m["miou"], 1.0)
        self.assertAlmostEqual(m["oa"], 1.0)


if __name__ == "__main__":
    unittest.main()
