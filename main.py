from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="MLOps pipeline API for Sentiment Analysis (Local Server)",
    version="1.0.0"
)

# Load sentiment analysis model at startup
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: list[str]


@app.post("/analyze/sentiment/")
async def analyze_sentiment(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = sentiment_analyzer(input.text)[0]
        return {
            "text": input.text,
            "label": result["label"],
            "confidence": round(result["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/analyze/sentiment/batch/")
async def analyze_sentiment_batch(input: BatchTextInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    try:
        results = sentiment_analyzer(input.texts)
        return {
            "results": [
                {
                    "text": text,
                    "label": result["label"],
                    "confidence": round(result["score"], 4)
                }
                for text, result in zip(input.texts, results)
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/health/")
def health():
    return {
        "status": "healthy",
        "models_loaded": ["sentiment-analysis"],
        "version": "1.0.0"
    }


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API v1.0 — POST /analyze/sentiment/ to analyze text."}
