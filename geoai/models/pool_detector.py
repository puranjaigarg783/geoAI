from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from .utils.config import PRITHVI_MODEL_NAME

class PoolDetector:
    def __init__(self):
        self.model = AutoModelForImageClassification.from_pretrained(PRITHVI_MODEL_NAME)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(PRITHVI_MODEL_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def detect_pools(self, images):
        """
        Detects pools in the given images.
        """
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions

