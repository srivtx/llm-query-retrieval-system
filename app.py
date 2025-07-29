"""
Railway deployment entry point for LLM-Powered Intelligent Query-Retrieval System
"""
from main import app
import os

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
