import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class SegmentationAgent:
    """
    Transfer-learning gastrointestinal segmentation agent.

    Uses a MobileNetV3-Large + DeepLabV3 segmentation model fine-tuned
    on Kvasir-SEG. The trained checkpoint is required.
    """

    CHECKPOINT = Path("outputs/trained_segmentation/deeplabv3_mobilenetv3_kvasir.pth")
    IMAGE_SIZE = 256

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path or self.CHECKPOINT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Trained segmentation checkpoint not found: {self.checkpoint_path}. "
                "Run train_segmentation.py first."
            )

        self.model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
        self.model.classifier = DeepLabHead(960, 2)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def segment(self, image_path, mask_path=None):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input image could not be loaded.")

        original_height, original_width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.IMAGE_SIZE, self.IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1)
        ).unsqueeze(0).float()

        with torch.no_grad():
            logits = self.model(tensor)["out"]
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        predicted_mask = (prediction * 255).astype(np.uint8)

        predicted_mask = cv2.resize(
            predicted_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return {
            "image": image,
            "mask": predicted_mask
        }

if __name__ == "__main__":
    print("Transfer-learning Segmentation Agent loaded successfully.")
