import os
import cv2

from agents.triage_agent import TriageAgent
from segmentation_agent_trained import SegmentationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reporting_agent import ReportingAgent


# Dataset paths
IMAGE_DIR = r"Kvasir-SEG\Kvasir-SEG\images"
MASK_DIR = r"Kvasir-SEG\Kvasir-SEG\masks"


def run_pipeline():

    # Select first image from dataset
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("No images found.")
        return

    image_name = image_files[0]

    image_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, image_name)

    print("\n========================================")
    print("   DIAGNOSTIC TRIAGE AGENT TEAM")
    print("========================================")

    print(f"\nProcessing image: {image_name}")

    # -----------------------------------
    # 1. TRIAGE AGENT
    # -----------------------------------

    image = cv2.imread(image_path)

    triage_agent = TriageAgent()
    triage_result = triage_agent.assess(image)

    print("\n[1] TRIAGE AGENT")
    print(triage_result)

    if triage_result["status"] == "rejected":
        print("\nImage rejected.")
        return

    # -----------------------------------
    # 2. SEGMENTATION AGENT
    # -----------------------------------

    segmentation_agent = SegmentationAgent()

    segmentation_result = segmentation_agent.segment(
        image_path,
        mask_path
    )

    predicted_mask = segmentation_result["mask"]

    print("\n[2] SEGMENTATION AGENT")
    print("Segmentation completed successfully.")

    # -----------------------------------
    # 3. EVALUATION AGENT
    # -----------------------------------

    evaluation_agent = EvaluationAgent()

    ground_truth = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    dice = evaluation_agent.dice_score(
        predicted_mask,
        ground_truth
    )

    iou = evaluation_agent.iou_score(
        predicted_mask,
        ground_truth
    )

    print("\n[3] EVALUATION AGENT")
    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score  : {iou:.4f}")

    # -----------------------------------
    # 4. REPORTING AGENT
    # -----------------------------------

    reporting_agent = ReportingAgent()

    report = reporting_agent.generate_report(
        triage_result,
        dice,
        iou
    )

    print("\n[4] REPORTING AGENT")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n========================================")
    print("Pipeline completed successfully.")
    print("========================================")


if __name__ == "__main__":
    run_pipeline()
