from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os

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

# Serve React frontend if static files exist (built by Docker multi-stage)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    # Serve Vite's built assets
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static-assets")

    @app.get("/{full_path:path}", tags=["Frontend"])
    async def serve_spa(full_path: str):
        file_path = os.path.join(static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    @app.get("/", tags=["Diagnostics"])
    def root():
        return {
            "message": "NLP Analysis API v3.0 (Enterprise Architecture)",
            "docs_url": "/docs"
        }
