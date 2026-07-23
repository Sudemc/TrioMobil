import unittest

import numpy as np

from scripts.compare_cams import fraction_of_top_pixels_inside_box


class TestBlankCamBoxMetrics(unittest.TestCase):
    def test_blank_cam_has_no_top_pixel_fraction(self):
        blank = np.zeros((4, 4), dtype=np.float32)

        fraction = fraction_of_top_pixels_inside_box(blank, (0, 0, 2, 2))

        self.assertIsNone(fraction)


if __name__ == "__main__":
    unittest.main()
