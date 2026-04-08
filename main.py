from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="MLOps pipeline API for Sentiment Analysis (Local Server)",
    version="1.0.0"
)

# Load sentiment analysis model at startup
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: list[str]


@app.post("/predict/")
async def predict_sentiment(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = classifier(input.text)[0]
        return {
            "text": input.text,
            "label": result["label"],
            "confidence": round(result["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/predict/batch/")
async def predict_batch(input: BatchTextInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")

    try:
        results = classifier(input.texts)
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


@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running. POST /predict/ with {\"text\": \"your text\"} to analyze sentiment."}
