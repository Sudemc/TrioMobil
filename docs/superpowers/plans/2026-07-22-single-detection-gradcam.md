# Single Detection Grad-CAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a repeatable Grad-CAM command that explains the class score of one selected YOLOv7 detection.

**Architecture:** Keep raw-candidate matching as pure functions with unit tests. The CLI runs NMS to select one final box, maps it to one raw YOLO candidate, and passes that candidate's class score to GradCAM through a raw-output model wrapper.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu124, YOLOv7, grad-cam 1.4.8, OpenCV, unittest.

## Global Constraints

- Never modify `third_party/`.
- Use `models/yolov7.pt`, `cuda:0`, `GradCAM`, class-score target, and layers `[102, 103, 104]`.
- Use `model.eval()` while retaining gradients for CAM.
- Do not hook layer 105 (`Detect`).
- Defaults: confidence threshold `0.25`, detection index `0`, visible box, no box-only renormalization.

---

### Task 1: Add and test raw-candidate matching helpers

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/yolov7_cam_utils.py`
- Create: `tests/test_yolov7_cam_utils.py`

**Interfaces:**
- `xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor`
- `box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor`
- `select_detection(detections: torch.Tensor, index: int) -> torch.Tensor`
- `match_raw_candidate(raw_predictions: torch.Tensor, detection: torch.Tensor, min_iou: float = 0.50) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yolov7_cam_utils.py
import unittest
import torch
from scripts.yolov7_cam_utils import xywh_to_xyxy, box_iou, select_detection, match_raw_candidate

class TestCamUtils(unittest.TestCase):
    def test_xywh_to_xyxy(self):
        source = torch.tensor([[50., 40., 20., 10.]])
        expected = torch.tensor([[40., 35., 60., 45.]])
        self.assertTrue(torch.equal(xywh_to_xyxy(source), expected))

    def test_iou_of_identical_boxes_is_one(self):
        box = torch.tensor([[0., 0., 10., 10.]])
        self.assertEqual(box_iou(box, box).item(), 1.0)

    def test_detection_index_is_selected(self):
        detections = torch.tensor([[0., 0., 10., 10., .95, 17.], [20., 20., 30., 30., .8, 3.]])
        self.assertTrue(torch.equal(select_detection(detections, 1), detections[1]))

    def test_empty_detections_raise_index_error(self):
        with self.assertRaises(IndexError):
            select_detection(torch.empty((0, 6)), 0)

    def test_raw_candidate_matches_class_and_iou(self):
        raw = torch.tensor([
            [5., 5., 10., 10., .9, .05, .95],
            [25., 25., 10., 10., .99, .9, .1],
            [5., 5., 10., 10., .8, .1, .96],
        ])
        detection = torch.tensor([0., 0., 10., 10., .86, 1.])
        self.assertEqual(match_raw_candidate(raw, detection), 2)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify it fails**

Run: `python -m unittest tests/test_yolov7_cam_utils.py -v`

Expected: FAIL because `scripts.yolov7_cam_utils` does not exist.

- [ ] **Step 3: Add the minimal implementation**

```python
# scripts/yolov7_cam_utils.py
import torch

def xywh_to_xyxy(boxes):
    output = boxes.clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output

def box_iou(boxes1, boxes2):
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)

def select_detection(detections, index):
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape [N, 6]")
    if index < 0 or index >= len(detections):
        raise IndexError(f"detection index {index} is not available")
    return detections[index]

def match_raw_candidate(raw_predictions, detection, min_iou=0.50):
    class_id = int(detection[5].item())
    raw_boxes = xywh_to_xyxy(raw_predictions[:, :4])
    same_class = raw_predictions[:, 5:].argmax(dim=1) == class_id
    if not same_class.any():
        raise RuntimeError(f"no raw candidate predicts class {class_id}")
    ious = box_iou(raw_boxes[same_class], detection[:4].unsqueeze(0)).squeeze(1)
    local_index = int(ious.argmax().item())
    if float(ious[local_index]) < min_iou:
        raise RuntimeError("raw candidate IoU is below the matching threshold")
    return int(torch.where(same_class)[0][local_index])
```

Create empty `scripts/__init__.py`.

- [ ] **Step 4: Verify and commit**

Run: `python -m unittest tests/test_yolov7_cam_utils.py -v`

Expected: 5 tests PASS.

```powershell
git add scripts/__init__.py scripts/yolov7_cam_utils.py tests/test_yolov7_cam_utils.py
git commit -m "feat: add YOLOv7 detection matching helpers"
```

### Task 2: Create the single-detection Grad-CAM CLI

**Files:**
- Create: `scripts/yolov7_single_detection_cam.py`
- Test: `tests/test_yolov7_cam_utils.py`

**Interfaces:**
- Inputs: `--weights`, `--source`, `--output-dir`, `--detection-index`, `--conf-threshold`, `--device`.
- Outputs: `<output-dir>/heatmap.png` and `<output-dir>/metadata.json`.

- [ ] **Step 1: Verify the script is absent**

Run: `python scripts/yolov7_single_detection_cam.py --help`

Expected: FAIL because the file does not exist.

- [ ] **Step 2: Write the CLI**

```python
# scripts/yolov7_single_detection_cam.py
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "yolov7"))
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from scripts.yolov7_cam_utils import match_raw_candidate, select_detection

class RawPredictionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image):
        return self.model(image)[0]

class ClassScoreTarget:
    def __init__(self, raw_index, class_id):
        self.raw_index, self.class_id = raw_index, class_id
    def __call__(self, predictions):
        return predictions[0, self.raw_index, 5 + self.class_id]

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolov7.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detection-index", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=.25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()

def main():
    args = arguments()
    device, output = torch.device(args.device), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = attempt_load(str(ROOT / args.weights), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    wrapped = RawPredictionModel(model)
    source = cv2.imread(args.source)
    if source is None:
        raise FileNotFoundError(args.source)
    size = int(model.stride.max()) * 80
    image = letterbox(source, new_shape=size, auto=False)[0]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = wrapped(tensor)
        detections = non_max_suppression(raw.clone(), args.conf_threshold, .45)[0]
    selected = select_detection(detections, args.detection_index)
    raw_index = match_raw_candidate(raw[0], selected)
    class_id = int(selected[5].item())

    layers = [model.model[index] for index in [102, 103, 104]]
    cam = GradCAM(model=wrapped, target_layers=layers, use_cuda=device.type == "cuda")
    grayscale = cam(tensor, targets=[ClassScoreTarget(raw_index, class_id)])[0]
    overlay = cv2.cvtColor(show_cam_on_image(rgb, grayscale, use_rgb=True), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2, confidence, _ = selected.detach().cpu().tolist()
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{model.names[class_id]} {confidence:.2f}"
    cv2.putText(overlay, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.imwrite(str(output / "heatmap.png"), overlay)
    metadata = {
        "weights": args.weights, "source": args.source, "method": "GradCAM",
        "target_mode": "class", "target_layers": [102, 103, 104],
        "conf_threshold": args.conf_threshold, "detection_index": args.detection_index,
        "class_id": class_id, "class_name": model.names[class_id],
        "detection_confidence": confidence, "selected_box_xyxy": [x1, y1, x2, y2],
        "raw_candidate_index": raw_index, "renormalize": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the interface and unit-test regression**

```powershell
python scripts/yolov7_single_detection_cam.py --help
python -m unittest tests/test_yolov7_cam_utils.py -v
```

Expected: help lists all six options; 5 tests PASS.

- [ ] **Step 4: Run the first GPU experiment**

```powershell
python scripts/yolov7_single_detection_cam.py --source third_party/yolov7/inference/images/horses.jpg --output-dir outputs/day2_single_detection --detection-index 0 --conf-threshold 0.25 --device cuda:0
```

Expected: `heatmap.png` and `metadata.json` are created. Metadata names one selected horse detection, class target mode, and layers `[102, 103, 104]`.

- [ ] **Step 5: Commit**

```powershell
git add scripts/yolov7_single_detection_cam.py
git commit -m "feat: add single detection Grad-CAM pipeline"
```

### Task 3: Challenge and document the output

**Files:**
- Modify: `notes/day2_cam_basics.md`
- Verify: `outputs/day2_single_detection/heatmap.png`
- Verify: `outputs/day2_single_detection/metadata.json`

- [ ] **Step 1: Inspect exact experiment metadata**

Run: `Get-Content outputs/day2_single_detection/metadata.json`

Expected: method `GradCAM`, target mode `class`, layers `[102, 103, 104]`, and `renormalize: false`.

- [ ] **Step 2: Add a note record**

Append:

```md
## Ilk tek-detection Grad-CAM deneyi

- Girdi: `horses.jpg`
- Yontem: `GradCAM`
- Target: detection index [metadata detection_index], class [metadata class_name]
- Target layer'lar: `[102, 103, 104]`
- Confidence threshold: `0.25`
- Renormalize: `False`
- Cikti: `outputs/day2_single_detection/heatmap.png`

Gozlem: [Heatmap'in vurguladigi alanlari yaz.]

Sinirlilik: Heatmap, secilen class score ile iliskili alanlari gosterir; hatanin kesin nedenini tek basina kanitlamaz.
```

- [ ] **Step 3: Repeat on another final detection**

```powershell
python scripts/yolov7_single_detection_cam.py --source third_party/yolov7/inference/images/horses.jpg --output-dir outputs/day2_single_detection_index1 --detection-index 1 --conf-threshold 0.25 --device cuda:0
```

Expected: a second heatmap and metadata file exist; its selected box differs from index 0.

- [ ] **Step 4: Record the comparison and commit documentation**

Record whether each heatmap tracks its green selected box rather than another horse/background, and whether the interpretation remains plausible without box renormalization.

```powershell
git add notes/day2_cam_basics.md
git commit -m "docs: record first single detection Grad-CAM experiment"
```

