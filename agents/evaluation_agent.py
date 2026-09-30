import numpy as np


class EvaluationAgent:
    """
    Calculates segmentation performance using
    Dice Score and Intersection over Union (IoU).
    """

    def dice_score(self, predicted_mask, ground_truth):
        predicted = predicted_mask > 0
        actual = ground_truth > 0

        intersection = np.logical_and(predicted, actual).sum()

        dice = (2.0 * intersection) / (
            predicted.sum() + actual.sum() + 1e-8
        )

        return float(dice)

    def iou_score(self, predicted_mask, ground_truth):
        predicted = predicted_mask > 0
        actual = ground_truth > 0

        intersection = np.logical_and(predicted, actual).sum()
        union = np.logical_or(predicted, actual).sum()

        iou = intersection / (union + 1e-8)

        return float(iou)


if __name__ == "__main__":
    print("Evaluation Agent loaded successfully.")
