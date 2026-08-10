import sys
import os
import uvicorn

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if __name__ == "__main__":
    print("="*60)
    print("      STARTING FASTAPI BACKEND SERVER (YOLO ENGINE)")
    print("="*60)
    print("Listening at: http://localhost:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
