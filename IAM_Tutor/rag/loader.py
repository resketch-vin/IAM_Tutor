import yaml
import re
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from .config import DATA_DIR

class IAMLoader:
    """IAM Tutor 전용 지식 베이스 로더"""
    
    def __init__(self):
        # 마크다운 헤더 분할 규칙 설정 (#, ##, ###)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        # 피드백 반영: 청크 사이즈 최적화 (600~800)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100
        )

    def load_all_documents(self) -> List[Document]:
        """data 폴더 내의 모든 마크다운 파일을 읽고 분할하여 반환"""
        all_docs = []
        
        # 피드백 반영: 폴더 구조 변경 (technique, harmony)
        for sub_dir in ["technique", "harmony"]:
            target_path = DATA_DIR / sub_dir
            if not target_path.exists():
                continue
                
            for file_path in target_path.glob("*.md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # YAML 프론트매터 파싱
                    front_matter = {}
                    if content.startswith("---"):
                        parts = re.split(r"---", content, maxsplit=2)
                        if len(parts) >= 3:
                            try:
                                front_matter = yaml.safe_load(parts[1])
                                content = parts[2].strip()
                            except Exception as e:
                                print(f"Error parsing YAML in {file_path.name}: {e}")
                    
                    # 1. 헤더 기반 분할 (맥락 유지)
                    header_splits = self.header_splitter.split_text(content)
                    
                    # 2. 메타데이터에 정보 추가 및 너무 큰 청크 재분할
                    for doc in header_splits:
                        # 기본 메타데이터 주입
                        doc.metadata["source"] = file_path.name
                        doc.metadata["category"] = sub_dir
                        
                        # YAML에서 추출한 메타데이터 병합
                        if front_matter:
                            doc.metadata.update(front_matter)
                        
                        # 청크가 너무 크면 다시 분할
                        if len(doc.page_content) > 800:
                            splits = self.text_splitter.split_documents([doc])
                            all_docs.extend(splits)
                        else:
                            all_docs.append(doc)
                            
        return all_docs

if __name__ == "__main__":
    loader = IAMLoader()
    docs = loader.load_all_documents()
    print(f"Total chunks loaded: {len(docs)}")
    if docs:
        print(f"Sample metadata: {docs[0].metadata}")
