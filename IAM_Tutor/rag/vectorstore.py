from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional
from langchain_core.documents import Document
from .config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
import os
import time

class IAMVectorStore:
    """FAISS를 이용한 VectorDB 관리 클래스"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            task_type="retrieval_document" # RAG 최적화
        )
        self.vector_db = None

    def create_and_save_index(self, documents: List[Document], batch_size: int = 100):
        """문서 리스트로부터 새로운 FAISS 인덱스 생성 및 저장 (유료 버전 최적화)"""
        if not documents:
            print("No documents to index.")
            return

        print(f"총 {len(documents)}개의 청크를 {batch_size}개씩 처리합니다...")
        
        # 첫 번째 배치 처리
        self._process_batch_with_retry(documents[:batch_size], is_first=True)
        
        # 나머지 배치 처리
        for i in range(batch_size, len(documents), batch_size):
            print(f"🕒 처리 중... {min(i + batch_size, len(documents))}/{len(documents)}")
            # 유료 버전의 경우 429 에러가 발생하지 않는다면 굳이 긴 대기 시간이 필요 없습니다.
            # 필요에 따라 짧은 간격(예: 1~2초)을 둘 수 있으나, 여기서는 속도를 위해 제거합니다.
            batch = documents[i:i+batch_size]
            self._process_batch_with_retry(batch, is_first=False)
            
        self.vector_db.save_local(str(FAISS_INDEX_PATH))
        print(f"\n✅ VectorDB가 성공적으로 저장되었습니다: {FAISS_INDEX_PATH}")

    def _process_batch_with_retry(self, batch: List[Document], is_first: bool, max_retries: int = 3):
        """각 배치를 재시도 로직과 함께 처리"""
        for attempt in range(max_retries):
            try:
                if is_first and self.vector_db is None:
                    self.vector_db = FAISS.from_documents(batch, self.embeddings)
                else:
                    self.vector_db.add_documents(batch)
                return # 성공 시 반환
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 70 * (attempt + 1)
                    print(f"⚠️ 할당량 초과 발생. {wait_time}초 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e

    def load_index(self) -> bool:
        """기존에 저장된 FAISS 인덱스 로드"""
        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_db = FAISS.load_local(
                str(FAISS_INDEX_PATH), 
                self.embeddings,
                allow_dangerous_deserialization=True # 로컬 파일이므로 허용
            )
            return True
        return False

    def get_retriever(self, k: int = 3):
        """리트리버 객체 반환"""
        if not self.vector_db:
            if not self.load_index():
                raise ValueError("VectorDB index not found. Please create it first.")
        return self.vector_db.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    # 테스트용 코드
    vstore = IAMVectorStore()
    if vstore.load_index():
        print("Successfully loaded existing index.")
    else:
        print("No index found.")
