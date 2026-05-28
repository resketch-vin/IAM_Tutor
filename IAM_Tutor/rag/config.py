import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent

# 데이터 경로
DATA_DIR = ROOT_DIR / "data"
GUITAR_TIPS_DIR = DATA_DIR / "guitar_tips"
MUSIC_THEORY_DIR = DATA_DIR / "music_theory"

# VectorDB 경로
FAISS_INDEX_PATH = ROOT_DIR / "faiss_index"

# 모델 설정
LLM_MODEL_NAME = "gemini-flash-latest"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"

# RAG 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K_SEARCH = 3

# API Key 검증 (보안 파일이므로 존재 여부만 확인)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set in .env file.")
