from rag.loader import IAMLoader
from rag.vectorstore import IAMVectorStore
import os
from dotenv import load_dotenv

def main():
    print("🚀 IAM Tutor 지식 베이스 구축을 시작합니다...")
    
    # 1. 데이터 로딩
    loader = IAMLoader()
    print("📂 문서를 읽고 분할하는 중...")
    documents = loader.load_all_documents()
    print(f"✅ 총 {len(documents)}개의 지식 청크가 준비되었습니다.")
    
    # 2. VectorDB 생성 및 저장
    vstore = IAMVectorStore()
    print("🧠 벡터 임베딩 및 FAISS 인덱스 생성 중 (시간이 소요될 수 있습니다)...")
    vstore.create_and_save_index(documents)
    
    print("\n✨ 모든 작업이 완료되었습니다! 이제 에이전트가 지식을 사용할 수 있습니다.")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ 에러: .env 파일에 GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    else:
        main()
