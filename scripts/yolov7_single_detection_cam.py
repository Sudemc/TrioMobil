import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "yolov7"))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression

from scripts.yolov7_cam_utils import match_raw_candidate, select_detection


class RawPredictionModel(torch.nn.Module):
    """Expose YOLOv7's pre-NMS predictions to Grad-CAM."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        original_export_check = torch.onnx.is_in_onnx_export
        torch.onnx.is_in_onnx_export = lambda: True
        try:
            return self.model(image)[0]
        finally:
            torch.onnx.is_in_onnx_export = original_export_check


class ClassScoreTarget:
    """Select one raw candidate's score for its predicted class."""

    def __init__(self, raw_index: int, class_id: int):
        self.raw_index = raw_index
        self.class_id = class_id

    def __call__(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions[self.raw_index, 5 + self.class_id]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Grad-CAM heatmap for one final YOLOv7 detection."
    )
    parser.add_argument("--weights", default="models/yolov7.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detection-index", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = attempt_load(str(ROOT / args.weights), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    wrapped_model = RawPredictionModel(model)

    source = cv2.imread(args.source)
    if source is None:
        raise FileNotFoundError(args.source)
    image = letterbox(source, new_shape=args.img_size, auto=False)[0]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_predictions = wrapped_model(input_tensor)
        detections = non_max_suppression(
            raw_predictions.clone(), args.conf_threshold, 0.45
        )[0]

    selected = select_detection(detections, args.detection_index)
    raw_index = match_raw_candidate(raw_predictions[0], selected)
    class_id = int(selected[5].item())

    target_layers = [model.model[index] for index in [102, 103, 104]]
    cam = GradCAM(
        model=wrapped_model,
        target_layers=target_layers,
        use_cuda=device.type == "cuda",
    )
    grayscale_cam = cam(
        input_tensor,
        targets=[ClassScoreTarget(raw_index, class_id)],
    )[0]
    overlay = cv2.cvtColor(
        show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True),
        cv2.COLOR_RGB2BGR,
    )

    x1, y1, x2, y2, confidence, _ = selected.detach().cpu().tolist()
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f"{model.names[class_id]} {confidence:.2f}"
    cv2.putText(
        overlay,
        label,
        (int(x1), max(20, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output / "heatmap.png"), overlay)

    metadata = {
        "weights": args.weights,
        "source": args.source,
        "method": "GradCAM",
        "target_mode": "class",
        "target_layers": [102, 103, 104],
        "conf_threshold": args.conf_threshold,
        "detection_index": args.detection_index,
        "class_id": class_id,
        "class_name": model.names[class_id],
        "detection_confidence": confidence,
        "selected_box_xyxy": [x1, y1, x2, y2],
        "raw_candidate_index": raw_index,
        "renormalize": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
