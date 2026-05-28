import os
from langchain.tools import tool
from rag.vectorstore import IAMVectorStore
from rag.config import K_SEARCH, LLM_MODEL_NAME

# VectorStore 싱글톤 패턴처럼 관리
vstore = IAMVectorStore()

@tool
def analyze_chord_progression(progression: str, key: str = "C Major") -> str:
    """
    사용자가 입력한 코드 진행을 기능적 화성학(Tonic, Sub-Dominant, Dominant) 관점에서 분석합니다.
    입력 형식 예시: "C - G - Am - F", "key: G major"
    RAG 지식 베이스를 활용하여 각 코드의 역할과 진행의 정체를 설명합니다.
    """
    try:
        # 진행의 성격 파악을 위한 RAG 검색 (기능론, 스케일 등)
        query = f"{key} 키에서 {progression} 진행의 화성학적 기능 분석"
        retriever = vstore.get_retriever(k=K_SEARCH)
        docs = retriever.invoke(query)
        
        context = "\n\n".join([f"[{doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}" for doc in docs])
        return f"[진행 분석 컨텍스트]\n{context}"
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

@tool
def detect_advanced_techniques(progression: str, key: str = "C Major") -> str:
    """
    코드 진행 내에서 세컨더리 도미넌트, 모달 인터체인지, 트라이톤 대리 등 고급 화성 기법을 탐지합니다.
    RAG 지식 베이스(화성학 07, 08번 등)를 참조하여 해당 기법의 효과와 원리를 설명합니다.
    """
    try:
        # 고급 기법 탐지를 위한 RAG 검색
        query = f"{key} 키 {progression} 진행에서의 세컨더리 도미넌트 및 모달 인터체인지 탐지"
        retriever = vstore.get_retriever(k=K_SEARCH)
        docs = retriever.invoke(query)
        
        context = "\n\n".join([f"[{doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}" for doc in docs])
        return f"[고급 기법 탐지 컨텍스트]\n{context}"
    except Exception as e:
        return f"탐지 중 오류 발생: {str(e)}"

@tool
def suggest_playing_guide(progression: str, user_level: int = 1) -> str:
    """
    분석된 코드 진행을 실제 기타로 연주하기 위한 가이드를 제공합니다.
    사용자 수준(1~5)에 맞춰 앵커 핑거, 약식 코드, 카포 위치 등을 제안합니다.
    """
    try:
        # 연주 가이드를 위한 RAG 검색 (기타팁 11, 13번 등)
        query = f"{progression} 코드 진행을 위한 {user_level}수준 연주 팁 및 약식 코드"
        retriever = vstore.get_retriever(k=K_SEARCH)
        docs = retriever.invoke(query)
        
        # 수준별 필터링 적용
        filtered_docs = [doc for doc in docs if user_level in doc.metadata.get("level", [1, 2, 3, 4, 5])]
        final_docs = filtered_docs if filtered_docs else docs
        
        context = "\n\n".join([f"[{doc.metadata.get('source', '알 수 없음')}]\n{doc.page_content}" for doc in final_docs])
        return f"[연주 가이드 컨텍스트]\n{context}"
    except Exception as e:
        return f"가이드 생성 중 오류 발생: {str(e)}"
