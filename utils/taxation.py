import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pdfplumber
from fastapi import UploadFile
from utils.vector_store import VectorStore
from config.settings import settings
from langchain.text_splitter import CharacterTextSplitter 
import json
from typing import List
import logging
from io import BytesIO

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxationService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)
        self.max_chunk_size = 65000  # 청크의 최대 크기 (바이트 기준)

    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """PDF 파일에서 텍스트를 추출"""
        text = ""
        with pdfplumber.open(BytesIO(pdf_data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def extract_text_from_xlsx(self, xlsx_data: bytes) -> str:
        """XLSX 파일에서 텍스트를 추출"""
        xlsx_df = pd.read_excel(xlsx_data)
        return xlsx_df.to_string(index=False)
    
    def split_text_for_milvus(self, text: str) -> List[str]:
        """텍스트를 Milvus의 최대 길이 이하로 단어 단위로 분할하는 함수"""
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

        return chunks

    def _store_chunks(self, text: str, content_data: dict, partition_name: str):
        """텍스트를 청크로 나누어 vectorDB에 저장"""
        chunks = self.split_text_for_milvus(text)
        for chunk_index, chunk in enumerate(chunks):
            content_data["chunk_index"] = chunk_index
            content_data["total_chunks"] = len(chunks)
            content_data["chunk"] = chunk.strip()
            self.vector_store.add_data_to_partitions(
                texts=content_data,
                partition_name=partition_name
            )
    
    def store_transaction_files(
            self, 
            transaction_files: List[bytes], 
            transaction_filenames: List[str], 
            business_id: int,
            partition_name : str
    ):
        """거래내역 파일을 처리하고 저장"""
        for file_data, file_name in zip(transaction_files, transaction_filenames):
            transaction_text = self.extract_text_from_xlsx(file_data)
            content_data = {
                "businessId": business_id,
                "fileName": file_name,
                "contentType": partition_name
            }
            self._store_chunks(transaction_text, content_data, partition_name)
            logger.info(f"거래내역 파일 {file_name} vectorDB 저장 완료")
    
    def store_income_tax_file(
            self, 
            income_tax_proof_file: bytes, 
            income_tax_proof_filename: str, 
            business_id: int,
            partition_name : str
    ):
        """소득/세액공제 파일을 처리하고 저장"""
        income_tax_text = self.extract_text_from_pdf(income_tax_proof_file)
        content_data = {
            "businessId": business_id,
            "fileName": income_tax_proof_filename,
            "contentType": partition_name
        }
        self._store_chunks(income_tax_text, content_data, partition_name)
        logger.info(f"소득/세액공제 파일 {income_tax_proof_filename} vectorDB 저장 완료")

    def store_questions(
            self, 
            answers: List[str], 
            business_id: int,
            partition_name : str
    ):
        """질문과 답변을 저장"""
        questions_map = {
            1: "현재 부양하고 있는 가족(배우자, 자녀, 부모 등)의 수",
            2: "부양가족 중 연간 소득이 100만 원을 초과하지 않는 가족의 수",
            3: "부양하는 각 자녀의 나이",
            4: "배우자의 연간소득이 100만원을 초과하는지 여부 (배우자가 없다면 없음이라고 적어주세요)",
            5: "부양가족 중 장애인으로 등록된 사람 수"
        }
        
        for i, answer in enumerate(answers, start=1):
            print(f"{i}번째 질문의 답변 : {answer}")
            question = questions_map[i]
            content_data = {
                "businessId": business_id,
                "questionNo": i,
                "contentType": partition_name,
                "chunk": f"{question} : {answer}"
            }
            # 각 질문과 답변을 개별적으로 저장
            self.vector_store.add_data_to_partitions(
                texts=[content_data],
                partition_name=partition_name
            )
            logger.info(f"질문과 답변 {partition_name} 저장 완료: {content_data}")
    
    def store_business_info(
            self, 
            business_id: int, 
            business_content: str, 
            business_type: str,
            partition_name : str
    ):
        """사업 정보를 저장"""
        content_data = {
            "businessId": business_id,
            "contentType": partition_name,
            "사업 내용": business_content,
            "사업자 유형": business_type
        }
        self.vector_store.add_data_to_partitions(
            texts=content_data,
            partition_name=partition_name
        )
        logger.info(f"사업 정보 {partition_name} 저장 완료")

    async def store_taxation(
            self,
            transaction_files: List[bytes],
            transaction_filenames: List[str],
            income_tax_proof_file : bytes,
            income_tax_proof_filename: str,
            answers: List[str],
            business_id: int,
            business_content: str,
            business_type: str
    ):
        """거래내역, 소득/세액공제 파일, 질문/답변, 사업 정보를 저장하는 메인 함수"""
        self.store_transaction_files(transaction_files, transaction_filenames, business_id, "TransactionFile")
        self.store_income_tax_file(income_tax_proof_file, income_tax_proof_filename, business_id, "IncomeTaxFile")
        self.store_questions(answers, business_id, "Question")
        self.store_business_info(business_id, business_content, business_type, "BusinessInfo")
