import sys
import unittest
from unittest.mock import patch

from scripts.yolov7_single_detection_cam import arguments


class TestTargetLayerArguments(unittest.TestCase):
    def test_default_target_layers_are_three_detection_inputs(self):
        argv = [
            "yolov7_single_detection_cam.py",
            "--source",
            "image.jpg",
            "--output-dir",
            "outputs/test",
        ]
        with patch.object(sys, "argv", argv):
            args = arguments()

        self.assertEqual(args.target_layers, [102, 103, 104])


if __name__ == "__main__":
    unittest.main()
