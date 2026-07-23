import unittest

import numpy as np

from scripts.compare_cams import compare_cam_arrays


class TestCompareCams(unittest.TestCase):
    def test_identical_cams_have_perfect_similarity(self):
        cam = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)

        metrics = compare_cam_arrays(cam, cam)

        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["top10_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
