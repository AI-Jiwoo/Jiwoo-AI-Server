import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import requests
from bs4 import BeautifulSoup
from utils.vector_store import VectorStore  # VectorDB와의 상호작용을 위한 사용자 정의 클래스
from config.settings import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaveTaxMethodUrl : 
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)


    # URL에서 텍스트 추출
    def extract_text_from_url(self, url):
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 모든 텍스트 추출
        text = soup.get_text(separator='\n', strip=True)
        return text

    # 추출된 텍스트와 URL을 VectorDB에 저장
    def save_tax_method_url(self):
        
        url = 'https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6596&cntntsId=7875'
    
        try:
            content = {
                "url": url,
                "description": "이 URL은 세액공제와 관련된 중요한 정보를 포함합니다."
            }
            
            partition_name = "TaxMethod"
            self.vector_store.add_data_to_partitions(content, partition_name=partition_name)
            logger.info("URL이 VectorDB에 저장되었습니다.")
        
        except Exception as e:
            logger.exception(f"데이터 저장 중 오류 발생: {str(e)}")

def main():
    
    save_tax_method_url = SaveTaxMethodUrl()

    save_tax_method_url.save_to_vectordb()

if __name__ == "__main__":
    main()
