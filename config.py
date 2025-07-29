# Configuration for LLM-Powered Intelligent Query-Retrieval System
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys - No longer needed for free local models!
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBZmXw-70iC-xBpqSH183DdmgPNp8OBX6E")
    
    # Model Configuration - Free Local Models
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free SentenceTransformer model
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Free conversational model
    USE_LOCAL_LLM = True  # Set to True to use free local LLM instead of Gemini
    
    # Document Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K_RESULTS = 5
    
    # API Configuration
    BEARER_TOKEN = "a85e6acfd3fc5388240c5d59b46de7129d843304f7c7bd1baa554ec4ff8ee0c5"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        issues = []
        
        if cls.GEMINI_API_KEY == "your-gemini-api-key-here":
            issues.append("GEMINI_API_KEY not set")
        
        if issues:
            print("⚠️  Configuration Issues:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n💡 To fix:")
            print("   1. Set environment variable: GEMINI_API_KEY=your_actual_key")
            print("   2. Or update config.py directly")
            return False
        
        print("✅ Configuration validated successfully")
        return True

# Environment setup
def setup_environment():
    """Setup environment variables"""
    print("🔧 Environment Setup:")
    print(f"   - Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"   - Chunk Size: {Config.CHUNK_SIZE}")
    print(f"   - Top K Results: {Config.TOP_K_RESULTS}")
    print(f"   - API Port: {Config.API_PORT}")
    
    return Config.validate_config()

if __name__ == "__main__":
    setup_environment()
