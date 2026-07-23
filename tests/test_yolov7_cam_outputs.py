import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.yolov7_single_detection_cam import save_grayscale_cam


class TestCamOutputs(unittest.TestCase):
    def test_grayscale_cam_is_saved_as_png_and_numpy_array(self):
        grayscale_cam = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_grayscale_cam(grayscale_cam, output)

            saved_array = np.load(output / "cam_grayscale.npy")
            self.assertTrue(np.array_equal(saved_array, grayscale_cam))
            self.assertTrue((output / "cam_grayscale.png").is_file())


if __name__ == "__main__":
    unittest.main()
