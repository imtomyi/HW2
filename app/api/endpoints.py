from fastapi import APIRouter, HTTPException
from app.schemas.nlp_schemas import TextInput, BatchTextInput, SummarizeInput
from app.services.ml_service import ml_service

router = APIRouter()

@router.post("/sentiment/")
async def analyze_sentiment(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = ml_service.sentiment_analyzer(input.text)[0]
        return {
            "text": input.text,
            "label": result["label"],
            "confidence": round(result["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/sentiment/batch/")
async def analyze_sentiment_batch(input: BatchTextInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")
    try:
        results = ml_service.sentiment_analyzer(input.texts)
        return {
            "results": [
                {"text": text, "label": r["label"], "confidence": round(r["score"], 4)}
                for text, r in zip(input.texts, results)
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/ner/")
async def analyze_ner(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        results = ml_service.ner_analyzer(input.text)
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

@router.post("/summarize/")
async def analyze_summarize(input: SummarizeInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(input.text.split()) < 30:
        raise HTTPException(status_code=400, detail="Text is too short to summarize. Provide at least 30 words.")
    try:
        result = ml_service.summarizer(input.text, max_length=input.max_length, min_length=input.min_length, do_sample=False)[0]
        return {
            "original_length": len(input.text.split()),
            "summary": result["summary_text"],
            "summary_length": len(result["summary_text"].split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/all/")
async def analyze_all(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        sent_result = ml_service.sentiment_analyzer(input.text)[0]
        ner_results = ml_service.ner_analyzer(input.text)
        entities = [
            {"entity": r["entity_group"], "word": r["word"], "confidence": round(r["score"], 4)}
            for r in ner_results
        ]
        summary = None
        if len(input.text.split()) >= 30:
            sum_result = ml_service.summarizer(input.text, max_length=130, min_length=30, do_sample=False)[0]
            summary = sum_result["summary_text"]

        return {
            "text": input.text,
            "sentiment": {"label": sent_result["label"], "confidence": round(sent_result["score"], 4)},
            "entities": entities,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
