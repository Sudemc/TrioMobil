import unittest

import torch

from scripts.yolov7_cam_utils import (
    box_iou,
    match_raw_candidate,
    select_detection,
    xywh_to_xyxy,
)


class TestCamUtils(unittest.TestCase):
    def test_xywh_to_xyxy(self):
        source = torch.tensor([[50.0, 40.0, 20.0, 10.0]])
        expected = torch.tensor([[40.0, 35.0, 60.0, 45.0]])
        self.assertTrue(torch.equal(xywh_to_xyxy(source), expected))

    def test_iou_of_identical_boxes_is_one(self):
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.assertEqual(box_iou(box, box).item(), 1.0)

    def test_detection_index_is_selected(self):
        detections = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0, 0.95, 17.0], [20.0, 20.0, 30.0, 30.0, 0.8, 3.0]]
        )
        self.assertTrue(torch.equal(select_detection(detections, 1), detections[1]))

    def test_empty_detections_raise_index_error(self):
        with self.assertRaises(IndexError):
            select_detection(torch.empty((0, 6)), 0)

    def test_raw_candidate_matches_nms_confidence(self):
        raw = torch.tensor(
            [
                [5.0, 5.0, 10.0, 10.0, 0.9, 0.05, 0.95],
                [25.0, 25.0, 10.0, 10.0, 0.99, 0.9, 0.1],
                [5.0, 5.0, 10.0, 10.0, 0.8, 0.1, 0.96],
            ]
        )
        detection = torch.tensor([0.0, 0.0, 10.0, 10.0, 0.86, 1.0])
        self.assertEqual(match_raw_candidate(raw, detection), 0)


if __name__ == "__main__":
    unittest.main()
