import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.sentiment_analyzer = None
        self.ner_analyzer = None
        self.zero_shot = None

    def load_models(self):
        """Loads Hugging Face models into memory."""
        logger.info("Loading ML models. This may take a moment...")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.ner_analyzer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        self.zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        logger.info("ML models loaded successfully.")

    def clear_models(self):
        """Releases memory by clearing model references."""
        logger.info("Clearing ML models from memory.")
        self.sentiment_analyzer = None
        self.ner_analyzer = None
        self.zero_shot = None

ml_service = MLService()
