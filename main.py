from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="NLP Analysis API",
    description="MLOps pipeline API for Multi-Feature NLP Analysis (Local Server)",
    version="2.0.0"
)

# Load models at startup
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_analyzer = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: list[str]


class SummarizeInput(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30


# --- Sentiment Analysis ---
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
                {"text": text, "label": r["label"], "confidence": round(r["score"], 4)}
                for text, r in zip(input.texts, results)
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- Named Entity Recognition ---
@app.post("/analyze/ner/")
async def analyze_ner(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        results = ner_analyzer(input.text)
        entities = [
            {
                "entity": r["entity_group"],
                "word": r["word"],
                "confidence": round(r["score"], 4),
                "start": r["start"],
                "end": r["end"]
            }
            for r in results
        ]
        return {"text": input.text, "entities": entities, "count": len(entities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- Text Summarization ---
@app.post("/analyze/summarize/")
async def analyze_summarize(input: SummarizeInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(input.text.split()) < 30:
        raise HTTPException(status_code=400, detail="Text is too short to summarize. Provide at least 30 words.")

    try:
        result = summarizer(input.text, max_length=input.max_length, min_length=input.min_length, do_sample=False)[0]
        return {
            "original_length": len(input.text.split()),
            "summary": result["summary_text"],
            "summary_length": len(result["summary_text"].split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- Combined Analysis ---
@app.post("/analyze/all/")
async def analyze_all(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # Sentiment
        sent_result = sentiment_analyzer(input.text)[0]

        # NER
        ner_results = ner_analyzer(input.text)
        entities = [
            {"entity": r["entity_group"], "word": r["word"], "confidence": round(r["score"], 4)}
            for r in ner_results
        ]

        # Summarization (only if text is long enough)
        summary = None
        if len(input.text.split()) >= 30:
            sum_result = summarizer(input.text, max_length=130, min_length=30, do_sample=False)[0]
            summary = sum_result["summary_text"]

        return {
            "text": input.text,
            "sentiment": {"label": sent_result["label"], "confidence": round(sent_result["score"], 4)},
            "entities": entities,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/health/")
def health():
    return {
        "status": "healthy",
        "models_loaded": ["sentiment-analysis", "ner", "summarization"],
        "version": "2.0.0"
    }


@app.get("/")
def root():
    return {
        "message": "NLP Analysis API v2.0",
        "endpoints": [
            "POST /analyze/sentiment/",
            "POST /analyze/ner/",
            "POST /analyze/summarize/",
            "POST /analyze/all/",
            "GET /health/"
        ]
    }
