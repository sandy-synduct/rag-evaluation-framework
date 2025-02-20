import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the root directory dynamically (assume `config.py` is inside `app/`)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PUBMED_EMAIL = os.getenv("PUBMED_EMAIL")
    
    # Ensure paths correctly point to files outside `app/`
    FAISS_INDEX_PATH = os.path.join(ROOT_DIR, os.getenv("FAISS_INDEX_PATH", "guidelines.index"))
    DOC_INDICES_PATH = os.path.join(ROOT_DIR, os.getenv("DOC_INDICES_PATH", "doc_indices.npy"))
    GUIDELINES_JSON = os.path.join(ROOT_DIR, os.getenv("GUIDELINES_JSON", "guidelines_database.json"))

config = Config()