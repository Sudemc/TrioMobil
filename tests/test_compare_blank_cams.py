import unittest

import numpy as np

from scripts.compare_cams import compare_cam_arrays


class TestCompareBlankCams(unittest.TestCase):
    def test_blank_cams_report_undefined_similarity(self):
        blank = np.zeros((2, 2), dtype=np.float32)

        metrics = compare_cam_arrays(blank, blank)

        self.assertIsNone(metrics["pearson"])
        self.assertIsNone(metrics["cosine"])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertIsNone(metrics["top10_jaccard"])


if __name__ == "__main__":
    unittest.main()
