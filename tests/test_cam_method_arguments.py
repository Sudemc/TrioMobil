import sys
import unittest
from unittest.mock import patch

from scripts.yolov7_single_detection_cam import arguments


class TestCamMethodArguments(unittest.TestCase):
    def test_default_method_is_gradcam(self):
        argv = [
            "yolov7_single_detection_cam.py",
            "--source",
            "image.jpg",
            "--output-dir",
            "outputs/test",
        ]
        with patch.object(sys, "argv", argv):
            args = arguments()

        self.assertEqual(args.method, "GradCAM")


if __name__ == "__main__":
    unittest.main()
