import cv2
import numpy as np
from pathlib import Path

from agents.segmentation_agent import SegmentationAgent


DATASET_ROOT = Path("Kvasir-SEG/Kvasir-SEG")
OUTPUT_DIR = Path("outputs/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_path = next((DATASET_ROOT / "images").glob("*.jpg"))

agent = SegmentationAgent()

result = agent.segment(str(image_path))

image = result["image"]
predicted_mask = result["mask"]

# Create colored overlay
overlay = image.copy()

overlay[predicted_mask > 0] = (
    0.5 * overlay[predicted_mask > 0]
    + 0.5 * np.array([0, 255, 0])
).astype(np.uint8)

# Save outputs
cv2.imwrite(
    str(OUTPUT_DIR / "original_image.jpg"),
    image
)

cv2.imwrite(
    str(OUTPUT_DIR / "predicted_mask.jpg"),
    predicted_mask
)

cv2.imwrite(
    str(OUTPUT_DIR / "prediction_overlay.jpg"),
    overlay
)

print("Visualization completed successfully.")
print("Original :", OUTPUT_DIR / "original_image.jpg")
print("Mask     :", OUTPUT_DIR / "predicted_mask.jpg")
print("Overlay  :", OUTPUT_DIR / "prediction_overlay.jpg")