"""
Vercel deployment entry point for LLM-Powered Intelligent Query-Retrieval System
"""
from main import app

# Export the FastAPI app for Vercel
# Vercel expects the app to be available as a callable
handler = app

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
