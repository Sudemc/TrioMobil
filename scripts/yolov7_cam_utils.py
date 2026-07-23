import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    output = boxes.clone()
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return output


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


def select_detection(detections: torch.Tensor, index: int) -> torch.Tensor:
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape [N, 6]")
    if index < 0 or index >= len(detections):
        raise IndexError(f"detection index {index} is not available")
    return detections[index]


def match_raw_candidate(
    raw_predictions: torch.Tensor,
    detection: torch.Tensor,
    min_iou: float = 0.50,
) -> int:
    class_id = int(detection[5].item())
    raw_boxes = xywh_to_xyxy(raw_predictions[:, :4])
    same_class = raw_predictions[:, 5:].argmax(dim=1) == class_id
    if not same_class.any():
        raise RuntimeError(f"no raw candidate predicts class {class_id}")

    candidate_indices = torch.where(same_class)[0]
    ious = box_iou(raw_boxes[candidate_indices], detection[:4].unsqueeze(0)).squeeze(1)
    eligible = ious >= min_iou
    if not eligible.any():
        raise RuntimeError("raw candidate IoU is below the matching threshold")

    candidate_indices = candidate_indices[eligible]
    candidate_scores = (
        raw_predictions[candidate_indices, 4]
        * raw_predictions[candidate_indices, 5 + class_id]
    )
    closest_score = (candidate_scores - detection[4]).abs().argmin()
    return int(candidate_indices[closest_score].item())
