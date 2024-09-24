from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image

class PoolDetector:
    def __init__(self):
        self.model = AutoModelForImageClassification.from_pretrained('models/pool_detector_model')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('models/pool_detector_model')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def detect_pools(self, images):
        """
        Detects pools in the given list of images.
        """
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()

