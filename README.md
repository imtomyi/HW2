# Age Prediction API

This is a simple FastAPI application for predicting age from uploaded facial images using the `deepface` library. The model is lightweight, making use of OpenCV for face detection by default.

## Project Structure
```
.
├── main.py            # The FastAPI application containing all routing and classification logic
├── requirements.txt   # File detailing the Python dependencies
└── README.md          # Documentation for the project
```

## Setup & Running

1. **Create and activate a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install the requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Testing the API**:
   - The interactive API documentation (Swagger UI) is available at: `http://127.0.0.1:8000/docs`
   - You can upload an image file using the Swagger UI page or via `curl`:
     ```bash
     curl -X POST "http://127.0.0.1:8000/predict_age/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
     ```
