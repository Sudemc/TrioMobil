import unittest

import numpy as np

from scripts.yolov7_single_detection_cam import renormalize_cam_within_box


class TestCamBoxRenormalization(unittest.TestCase):
    def test_only_box_region_is_scaled_to_zero_one(self):
        cam = np.array(
            [
                [9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 2.0, 9.0],
                [9.0, 3.0, 4.0, 9.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )

        result = renormalize_cam_within_box(cam, (1, 1, 3, 3))

        self.assertEqual(result[0, 0], 0.0)
        self.assertAlmostEqual(result[1:3, 1:3].min(), 0.0)
        self.assertAlmostEqual(result[1:3, 1:3].max(), 1.0)


if __name__ == "__main__":
    unittest.main()
