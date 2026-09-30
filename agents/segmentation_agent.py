import cv2
import numpy as np


class SegmentationAgent:
    """
    Lightweight image segmentation agent.

    Uses LAB color-space processing, adaptive thresholding,
    morphological filtering, and connected-component analysis.
    """

    def segment(self, image_path, mask_path=None):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Input image could not be loaded.")

        original_height, original_width = image.shape[:2]

        # Resize for faster CPU processing
        max_size = 512

        scale = min(
            max_size / original_width,
            max_size / original_height,
            1.0
        )

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        processed = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Convert to LAB color space
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)

        # Extract channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # CLAHE improves local contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_enhanced = clahe.apply(l_channel)

        # Combine color information
        enhanced = cv2.addWeighted(
            l_enhanced,
            0.6,
            a_channel,
            0.4,
            0
        )

        # Otsu threshold
        _, binary = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Try both foreground orientations
        candidates = [
            binary,
            cv2.bitwise_not(binary)
        ]

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)

        best_mask = candidates[0]
        best_score = -1

        for candidate in candidates:

            cleaned = cv2.morphologyEx(
                candidate,
                cv2.MORPH_OPEN,
                kernel_small
            )

            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                kernel_large,
                iterations=2
            )

            num_labels, labels, stats, _ = (
                cv2.connectedComponentsWithStats(
                    cleaned,
                    connectivity=8
                )
            )

            if num_labels <= 1:
                continue

            # Select meaningful connected regions
            areas = stats[1:, cv2.CC_STAT_AREA]

            largest_index = np.argmax(areas) + 1
            largest_area = stats[
                largest_index,
                cv2.CC_STAT_AREA
            ]

            image_area = cleaned.shape[0] * cleaned.shape[1]
            area_ratio = largest_area / image_area

            # Prefer regions of reasonable size
            if 0.01 <= area_ratio <= 0.60:

                score = area_ratio

                if score > best_score:
                    best_score = score
                    best_mask = (
                        labels == largest_index
                    ).astype(np.uint8) * 255

        # Final smoothing
        best_mask = cv2.morphologyEx(
            best_mask,
            cv2.MORPH_CLOSE,
            kernel_large,
            iterations=2
        )

        # Resize back to original dimensions
        predicted_mask = cv2.resize(
            best_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return {
            "image": image,
            "mask": predicted_mask
        }


if __name__ == "__main__":
    print("Improved Segmentation Agent loaded successfully.")