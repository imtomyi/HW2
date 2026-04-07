from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from deepface import DeepFace

app = FastAPI(
    title="Face Analysis API",
    description="A simple API to predict age and gender from uploaded face images using DeepFace.",
    version="2.0.0"
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image. Invalid format.")

        results = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender'],
            enforce_detection=True,
            detector_backend='opencv'
        )

        if isinstance(results, list):
            predictions = []
            for result in results:
                predictions.append({
                    "age": result.get("age"),
                    "gender": result.get("dominant_gender"),
                    "gender_confidence": result.get("gender", {}),
                    "face_confidence": result.get("face_confidence")
                })
            return {"results": predictions, "faces_detected": len(results)}
        else:
            return {
                "results": [{
                    "age": results.get("age"),
                    "gender": results.get("dominant_gender"),
                    "gender_confidence": results.get("gender", {}),
                    "face_confidence": results.get("face_confidence")
                }],
                "faces_detected": 1
            }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/predict_age/")
async def predict_age(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image. Invalid format.")

        results = DeepFace.analyze(
            img_path=img,
            actions=['age'],
            enforce_detection=True,
            detector_backend='opencv'
        )

        if isinstance(results, list):
            predictions = []
            for result in results:
                predictions.append({
                    "age": result.get("age"),
                    "face_confidence": result.get("face_confidence")
                })
            return {"results": predictions, "faces_detected": len(results)}
        else:
            return {"results": [{"age": results.get("age"), "face_confidence": results.get("face_confidence")}], "faces_detected": 1}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def root():
    return {"message": "Face Analysis API v2.0 - Age & Gender Prediction. POST /predict/ for both, or /predict_age/ for age only."}
