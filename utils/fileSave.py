import os
import sys
import pdfplumber

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import settings
from typing import List
import logging
from io import BytesIO
from utils.vector_store import VectorStore 
from langchain.text_splitter import  CharacterTextSplitter  

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxFileSave : 
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)
        self.max_chunk_size = 1000

    
    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """PDF에서 텍스트를 추출하는 함수"""
        text = ""
        with pdfplumber.open(BytesIO(pdf_data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text


    # 텍스트를 Milvus의 최대 길이 이하로 분할하는 함수
    def split_text_for_milvus(self, text: str) -> List[str]:
        """텍스트를 청크하는 함수"""
        chunks = []
        current_chunk = ""
        current_chunk_bytes = 0

        for word in text.split():
            word_bytes = (word + ' ').encode('utf-8')
            if current_chunk_bytes + len(word_bytes) > self.max_chunk_size:
                # 현재 청크가 최대 길이에 도달하면 저장하고 새 청크 시작
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_bytes = 0
            
            current_chunk += word + ' '
            current_chunk_bytes += len(word_bytes)
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 청크 크기 로깅
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} size: {len(chunk.encode('utf-8'))} bytes")
        
        return chunks


    # 세금 관련 파일을 VectorDB에 저장하는 함수
    def store_tax_documents(
            self,
            taxation_files : List[bytes],
            taxation_filename : List[str],
            partition_name : str
            ):
        """
        세금 관련 파일을 VectorDB에 저장하는 함수
        :param tax_files : 저장할 파일 내용
        :param tax_filename : 저장할 파일 제목
        """

        # text_splitter = CharacterTextSplitter(chunk_size=65000, chunk_overlap=0)
        try: 

            for file_data, file_name in zip(taxation_files, taxation_filename):
                file_text = self.extract_text_from_pdf(file_data)

                content_data = {
                    "fileName": file_name,
                    "contentType": partition_name
                }

                self._store_chunks(file_text, content_data, partition_name)
                logger.info(f"세금 관련 파일 {file_name} vectorDB 저장 완료")

        except Exception as e:
            print(f"파일 저장 중 오류 발생: {str(e)}")
            print("세금 관련 파일 저장이 완료되지 않았습니다.")


    def _store_chunks(self, text: str, content_data: dict, partition_name: str):
        """텍스트를 청크로 나누어 vectorDB에 저장"""
        chunks = self.split_text_for_milvus(text)
        for chunk_index, chunk in enumerate(chunks) :
            content_data["chunk_index"] = chunk_index
            content_data["total_chunks"] = len(chunks)
            content_data["chunk"] = chunk.strip()
            self.vector_store.add_data_to_partitions(
                texts=content_data,
                partition_name=partition_name
            )


    # VectorDB에 저장된 세금 관련 파일이 잘 저장되었는지 확인하는 함수
    def verify_stored_tax_documents(collection_name):
        """
        VectorDB에 저장된 세금 관련 파일이 잘 저장되었는지 확인하는 함수
        :param vector_store: VectorDB 인스턴스
        """

        vector_store = VectorStore()

        vector_store.use_custom_collection(collection_name)

        try:
            # 검색 쿼리와 메타데이터 필터를 이용해 저장된 문서를 검색
            # results = vector_store.search_in_all_partitions(query="tax_document or tax_files", k=10)
            results = vector_store.search_in_partition(query="", partition_name="TaxFiles", k=30)

            if results:
                for result in results:
                    print(f"Found document in {collection_name}: {result['content'][:200]}...")  # 첫 200자만 출력
            else:
                print("세금 관련 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"오류 발생: {str(e)}")

    
    # 세금 관련 파일 저장 
    async def save_Tax_Files(
        self,
        taxation_files : List[bytes],
        taxation_filename: List[str]
    ):
        """세액 관련 파일을 저장하는 메인 함수"""
        collection_name = settings.COLLECTION_NAME2
        self.store_tax_documents(taxation_files, taxation_filename, "TaxFiles")
        # self.verify_stored_tax_documents(collection_name)
        

