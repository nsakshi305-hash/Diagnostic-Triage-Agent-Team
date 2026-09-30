class ReportingAgent:
    """
    Generates a structured report from the
    triage and segmentation evaluation results.
    """

    def generate_report(self, triage_result, dice, iou):

        report = {
            "triage_status": triage_result["status"],
            "image_size": triage_result.get("image_size", "Unknown"),
            "dice_score": round(dice, 4),
            "iou_score": round(iou, 4)
        }

        if dice >= 0.90:
            report["segmentation_quality"] = "Excellent"
        elif dice >= 0.75:
            report["segmentation_quality"] = "Good"
        elif dice >= 0.50:
            report["segmentation_quality"] = "Moderate"
        else:
            report["segmentation_quality"] = "Low"

        report["note"] = (
            "This report is generated for research and educational purposes "
            "and is not a medical diagnosis."
        )

        return report


if __name__ == "__main__":
    print("Reporting Agent loaded successfully.")
