import cv2


class TriageAgent:
    """
    Performs basic input-image quality checks
    before segmentation.
    """

    def assess(self, image):
        if image is None:
            return {
                "status": "rejected",
                "reason": "Image could not be loaded."
            }

        height, width = image.shape[:2]

        if height < 200 or width < 200:
            return {
                "status": "rejected",
                "reason": "Image resolution is too small."
            }

        if len(image.shape) != 3:
            return {
                "status": "rejected",
                "reason": "Image must be a color image."
            }

        return {
            "status": "accepted",
            "reason": "Image passed the basic quality checks.",
            "image_size": f"{width} x {height}"
        }


if __name__ == "__main__":
    print("Triage Agent loaded successfully.")