import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class MedicalClassifierAgent:
    """Pretrained GI endoscopy finding classifier based on Kvasir v2."""

    MODEL_NAME = "mmuratarat/kvasir-v2-classifier"

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        top = torch.topk(probabilities, k=min(3, len(probabilities)))
        predictions = []
        for score, index in zip(top.values, top.indices):
            label = self.model.config.id2label.get(int(index), str(int(index)))
            predictions.append({"label": label, "confidence": float(score)})

        return predictions
