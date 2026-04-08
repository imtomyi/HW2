from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.services.ml_service import ml_service
from app.api.endpoints import router as nlp_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Lifespan context manager for robust loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute on startup
    logger.info("Starting up FastAPI application...")
    ml_service.load_models()
    yield
    # Execute on shutdown
    logger.info("Shutting down FastAPI application...")
    ml_service.clear_models()

app = FastAPI(
    title="NLP Analysis API",
    description="MLOps Pipeline API with Lifespan Management (Enterprise Ready)",
    version="3.0.0",
    lifespan=lifespan
)

# Added CORS middleware for the frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nlp_router, prefix="/analyze", tags=["NLP Analysis"])

@app.get("/health/", tags=["Diagnostics"])
def health():
    return {
        "status": "healthy",
        "models_loaded": ml_service.sentiment_analyzer is not None,
        "version": "3.0.0"
    }

@app.get("/", tags=["Diagnostics"])
def root():
    return {
        "message": "NLP Analysis API v3.0 (Enterprise Architecture)",
        "docs_url": "/docs"
    }
