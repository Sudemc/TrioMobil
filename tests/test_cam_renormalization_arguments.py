import sys
import unittest
from unittest.mock import patch

from scripts.yolov7_single_detection_cam import arguments


class TestCamRenormalizationArguments(unittest.TestCase):
    def test_box_renormalization_is_disabled_by_default(self):
        argv = [
            "yolov7_single_detection_cam.py",
            "--source",
            "image.jpg",
            "--output-dir",
            "outputs/test",
        ]
        with patch.object(sys, "argv", argv):
            args = arguments()

        self.assertFalse(args.renormalize_within_box)


if __name__ == "__main__":
    unittest.main()
