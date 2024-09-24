from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from datasets import load_dataset
import torch
from utils.config import PRITHVI_MODEL_NAME

def fine_tune_model():
    """
    Fine-tunes the Prithvi model to detect pools.
    """
    dataset = load_dataset('your-pool-dataset')  # Replace with actual dataset

    model = AutoModelForImageClassification.from_pretrained(PRITHVI_MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(PRITHVI_MODEL_NAME)

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples['image']]
        inputs = feature_extractor(images=images, return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs

    dataset = dataset.map(preprocess, batched=True)

    model.save_pretrained('models/pool_detector_model')
    feature_extractor.save_pretrained('models/pool_detector_model')

