import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import requests
from bs4 import BeautifulSoup
from vector_store import VectorStore  # VectorDB와의 상호작용을 위한 사용자 정의 클래스
from config.settings import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 추출된 텍스트와 URL을 VectorDB에 저장
def save_to_vectordb(url):
    try:
        vector_store = VectorStore()
        vector_store.use_custom_collection(settings.COLLECTION_NAME2)
        
        content = {
            "url": url,
            "description" : "간편장부 컬럼",
            "columns" : ['일자', '계정과목', '거래내용', '거래처', '수입(매출) 금액', '수입(매출) 부가세', '비용(원가관련 매입포함) 금액', '비용(원가관련 매입포함) 부가세', '사업용 유형자산 및 무형자 산 증감(매매) 금액', '사업용 유형자산 및 무형자산 증감(매매) 부가세', '비고']
        }
        
        partition_name = "SimpleTransaction"
        vector_store.add_data_to_partitions(content, partition_name=partition_name)
        logger.info("간편장부 컬럼이 VectorDB에 저장되었습니다.")
    
    except Exception as e:
        logger.exception(f"데이터 저장 중 오류 발생: {str(e)}")

def main():
    url = "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2231&cntntsId=7670"

    # URL 저장
    save_to_vectordb(url)

if __name__ == "__main__":
    main()
