from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from deepface import DeepFace

app = FastAPI(
    title="Age Prediction API",
    description="A simple API to predict age from uploaded face images using DeepFace.",
    version="1.0.0"
)

@app.post("/predict_age/")
async def predict_age(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the image file as bytes
        contents = await file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode the image for OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image. Invalid format.")

        # Analyze the image for age using DeepFace
        # We use 'opencv' backend as it is lightweight for face detection
        results = DeepFace.analyze(
            img_path=img,
            actions=['age'],
            enforce_detection=True,
            detector_backend='opencv'
        )

        # DeepFace.analyze can return a list if multiple faces are detected
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
        # This occurs if no face could be detected in the image
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def root():
    return {"message": "Age Prediction API is running. Send a POST request to /predict_age/ with an image file."}
