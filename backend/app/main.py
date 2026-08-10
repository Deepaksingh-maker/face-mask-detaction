import sys
import os

app_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(app_dir)
root_dir = os.path.dirname(backend_dir)

for path in [root_dir, backend_dir, app_dir]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.api.mask_detection import router as mask_router
except ImportError:
    try:
        from app.api.mask_detection import router as mask_router
    except ImportError:
        from api.mask_detection import router as mask_router

app = FastAPI(
    title="AI Face Mask Detector API",
    version="1.0.0",
    description="Real-Time Deep Learning Face Mask Detection & Safety Status Backend Server"
)

# Enable CORS for NPM frontend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mask_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "system": "AI Face Mask Detection API Server",
        "status": "ONLINE",
        "endpoints": {
            "predict": "/api/v1/mask-detection/predict",
            "health": "/api/v1/mask-detection/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
