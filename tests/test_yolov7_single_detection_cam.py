import sys
import unittest
from unittest.mock import patch

from scripts.yolov7_single_detection_cam import arguments


class TestCamArguments(unittest.TestCase):
    def test_default_image_size_is_640(self):
        argv = [
            "yolov7_single_detection_cam.py",
            "--source",
            "image.jpg",
            "--output-dir",
            "outputs/test",
        ]
        with patch.object(sys, "argv", argv):
            args = arguments()

        self.assertEqual(args.img_size, 640)


if __name__ == "__main__":
    unittest.main()
