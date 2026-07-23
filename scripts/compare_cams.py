import argparse
import json
from pathlib import Path

import numpy as np


def compare_cam_arrays(cam_a: np.ndarray, cam_b: np.ndarray) -> dict[str, float]:
    if cam_a.shape != cam_b.shape:
        raise ValueError("CAM arrays must have the same shape")

    flat_a = cam_a.ravel()
    flat_b = cam_b.ravel()
    pearson = (
        None
        if np.std(flat_a) == 0 or np.std(flat_b) == 0
        else float(np.corrcoef(flat_a, flat_b)[0, 1])
    )
    norm_product = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    cosine = (
        None if norm_product == 0 else float(np.dot(flat_a, flat_b) / norm_product)
    )
    mae = float(np.abs(cam_a - cam_b).mean())

    top_a = (cam_a > 0) & (cam_a >= np.quantile(cam_a, 0.90))
    top_b = (cam_b > 0) & (cam_b >= np.quantile(cam_b, 0.90))
    intersection = np.logical_and(top_a, top_b).sum()
    union = np.logical_or(top_a, top_b).sum()
    top10_jaccard = None if union == 0 else float(intersection / union)

    return {
        "pearson": pearson,
        "cosine": cosine,
        "mae": mae,
        "top10_jaccard": top10_jaccard,
    }


def box_coordinates(values: list[float]) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in values)


def mean_inside_box(cam: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(cam[y1:y2, x1:x2].mean())


def fraction_of_top_pixels_inside_box(
    cam: np.ndarray, box: tuple[int, int, int, int]
) -> float | None:
    top_pixels = (cam > 0) & (cam >= np.quantile(cam, 0.90))
    total_top_pixels = top_pixels.sum()
    if total_top_pixels == 0:
        return None
    x1, y1, x2, y2 = box
    return float(top_pixels[y1:y2, x1:x2].sum() / total_top_pixels)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two saved grayscale CAM arrays.")
    parser.add_argument("--cam-a-dir", required=True)
    parser.add_argument("--cam-b-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    directory_a = Path(args.cam_a_dir)
    directory_b = Path(args.cam_b_dir)
    cam_a = np.load(directory_a / "cam_grayscale.npy")
    cam_b = np.load(directory_b / "cam_grayscale.npy")
    metadata_a = json.loads((directory_a / "metadata.json").read_text(encoding="utf-8"))
    metadata_b = json.loads((directory_b / "metadata.json").read_text(encoding="utf-8"))
    box_a = box_coordinates(metadata_a["selected_box_xyxy"])
    box_b = box_coordinates(metadata_b["selected_box_xyxy"])

    metrics = compare_cam_arrays(cam_a, cam_b)
    metrics.update(
        {
            "cam_a_detection_index": metadata_a["detection_index"],
            "cam_b_detection_index": metadata_b["detection_index"],
            "cam_a_mean_own_box": mean_inside_box(cam_a, box_a),
            "cam_a_mean_other_box": mean_inside_box(cam_a, box_b),
            "cam_b_mean_own_box": mean_inside_box(cam_b, box_b),
            "cam_b_mean_other_box": mean_inside_box(cam_b, box_a),
            "cam_a_top10_fraction_in_own_box": fraction_of_top_pixels_inside_box(
                cam_a, box_a
            ),
            "cam_b_top10_fraction_in_own_box": fraction_of_top_pixels_inside_box(
                cam_b, box_b
            ),
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
