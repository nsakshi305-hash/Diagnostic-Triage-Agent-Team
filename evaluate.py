from pathlib import Path
import csv
import json
import cv2
import numpy as np

from agents.segmentation_agent import SegmentationAgent
from agents.evaluation_agent import EvaluationAgent


DATASET_ROOT = Path("Kvasir-SEG/Kvasir-SEG")
OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 50

images = sorted((DATASET_ROOT / "images").glob("*.jpg"))[:MAX_IMAGES]

segmentation_agent = SegmentationAgent()
evaluation_agent = EvaluationAgent()

results = []

for image_path in images:

    mask_path = DATASET_ROOT / "masks" / image_path.name

    result = segmentation_agent.segment(str(image_path))

    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    dice = evaluation_agent.dice_score(
        result["mask"],
        ground_truth
    )

    iou = evaluation_agent.iou_score(
        result["mask"],
        ground_truth
    )

    results.append({
        "image": image_path.name,
        "dice": round(dice, 4),
        "iou": round(iou, 4)
    })


dice_scores = [r["dice"] for r in results]
iou_scores = [r["iou"] for r in results]

summary = {
    "images_evaluated": len(results),
    "mean_dice": round(float(np.mean(dice_scores)), 4),
    "median_dice": round(float(np.median(dice_scores)), 4),
    "best_dice": round(float(np.max(dice_scores)), 4),
    "worst_dice": round(float(np.min(dice_scores)), 4),
    "mean_iou": round(float(np.mean(iou_scores)), 4),
    "median_iou": round(float(np.median(iou_scores)), 4),
    "best_iou": round(float(np.max(iou_scores)), 4),
    "worst_iou": round(float(np.min(iou_scores)), 4)
}


csv_path = OUTPUT_DIR / "segmentation_results.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["image", "dice", "iou"]
    )

    writer.writeheader()
    writer.writerows(results)


json_path = OUTPUT_DIR / "evaluation_summary.json"

with open(json_path, "w") as file:
    json.dump(summary, file, indent=4)


print("=" * 50)
print("SEGMENTATION EVALUATION")
print("=" * 50)

for key, value in summary.items():
    print(f"{key}: {value}")

print("=" * 50)
print(f"CSV saved : {csv_path}")
print(f"JSON saved: {json_path}")