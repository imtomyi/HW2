from fastapi import APIRouter, HTTPException
from app.schemas.nlp_schemas import TextInput, BatchTextInput, ClassifyInput
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

@router.post("/classify/")
async def analyze_classify(input: ClassifyInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not input.labels:
        raise HTTPException(status_code=400, detail="Labels list cannot be empty.")
    try:
        result = ml_service.zero_shot(input.text, candidate_labels=input.labels)
        classifications = [
            {"label": label, "confidence": round(score, 4)}
            for label, score in zip(result["labels"], result["scores"])
        ]
        return {
            "text": input.text,
            "top_label": result["labels"][0],
            "classifications": classifications
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/all/")
async def analyze_all(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        # Sentiment
        sent_result = ml_service.sentiment_analyzer(input.text)[0]

        # NER
        ner_results = ml_service.ner_analyzer(input.text)
        entities = [
            {"entity": r["entity_group"], "word": r["word"], "confidence": round(r["score"], 4)}
            for r in ner_results
        ]

        # Zero-shot classification
        default_labels = ["politics", "technology", "sports", "entertainment", "business", "science", "health"]
        classify_result = ml_service.zero_shot(input.text, candidate_labels=default_labels)
        classifications = [
            {"label": label, "confidence": round(score, 4)}
            for label, score in zip(classify_result["labels"], classify_result["scores"])
        ]

        return {
            "text": input.text,
            "sentiment": {"label": sent_result["label"], "confidence": round(sent_result["score"], 4)},
            "entities": entities,
            "classification": {"top_label": classify_result["labels"][0], "all": classifications}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
