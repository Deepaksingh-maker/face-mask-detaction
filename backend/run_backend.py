"""
Launcher for Python FastAPI AI Backend Server inside backend folder.
Run from inside backend folder:
    python run_backend.py
"""
import sys
import os
import uvicorn

backend_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(backend_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    print("🚀 Starting AI Face Mask Detector FastAPI Backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
