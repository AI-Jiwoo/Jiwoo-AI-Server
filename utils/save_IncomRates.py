import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from utils.vector_store import VectorStore
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re


from config.settings import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaveIncomeRate : 
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)

    def fetch_html(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            logger.error(f"페이지를 가져오는 중 에러가 발생했습니다. 상태 코드: {response.status_code}")
            return None


    def extract_years_from_text(self, text):
        # 연도 범위를 처리하기 위한 정규 표현식
        range_pattern = re.compile(r'\((\d{4})~(\d{4})년 귀속\)')
        single_pattern = re.compile(r'\((\d{4})년 귀속\)')
        
        match = range_pattern.search(text)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            return list(range(start_year, end_year + 1))
        
        match = single_pattern.search(text)
        if match:
            return [int(match.group(1))]
        
        return []

    def extract_year_table_pairs(self,soup):
        year_table_data = []
        h3_tags = soup.find_all('h3', class_='tit1')
        
        for h3 in h3_tags:
            years = self.extract_years_from_text(h3.text.strip())
            if years:
                table = h3.find_next('table')
                if table:
                    year_table_data.append((years, table))
                    logger.info(f"{years}년도의 테이블을 찾았습니다.")
                else:
                    logger.warning(f"{years}년도의 테이블을 찾을 수 없습니다.")
        
        return year_table_data

    def extract_table_data(self, years, table):
        table_data = []
        rows = table.find_all('tr')
        
        start_index = 1 if rows[0].find_all('th') else 0
        
        for row in rows[start_index:]:
            columns = row.find_all(['td', 'th'])
            if len(columns) >= 3:
                tax_standard = columns[0].text.strip()
                tax_rate = columns[1].text.strip()
                tax_credit = columns[2].text.strip()
                
                for year in years:
                    entry = {
                        "category": "종합소득세",
                        "year": year,
                        "income_range": tax_standard,
                        "tax_rate": tax_rate,
                        "deduction": tax_credit
                    }
                    table_data.append(entry)
            else:
                logger.warning(f"잘못된 형식의 행을 발견했습니다: {row}")
        
        return table_data

    def find_most_recent_year_data(self, all_tax_data, query_year):
        available_years = sorted({data["year"] for data in all_tax_data})
        logger.info(f"데이터에 포함된 연도들: {available_years}")

        if query_year not in available_years:
            logger.info(f"{query_year}년에 대한 정보가 없습니다. 가장 최근의 정보로 대체합니다.")
            most_recent_year = available_years[-1]  # 가장 최근 연도
            return [data for data in all_tax_data if data["year"] == most_recent_year]
        
        return [data for data in all_tax_data if data["year"] == query_year]

    def get_existing_tax_data(self):
        """
        기존의 종합소득세 데이터 가져오기
        """

        return self.vector_store.search_in_partition(query="", partition_name="IncomeRates", k=1000)

    def are_records_equal(self, existing_record, new_record):
        """
        기존 레코드와 새 레코드를 비교하여 동일한지 확인합니다.
        동일하면 True, 그렇지 않으면 False를 반환합니다.
        """
        # 기존 레코드의 content는 JSON 문자열로 저장되어 있으므로, 이를 파싱하여 비교합니다.
        existing_data = json.loads(existing_record['content'])

        # 새 레코드도 동일한 구조로 변환합니다.
        new_data = {
            "연도": new_record["year"],
            "과세표준": new_record["income_range"],
            "세율": new_record["tax_rate"],
            "누진공세": new_record["deduction"]
        }

        # 모든 필드를 비교합니다.
        return (
            existing_data["연도"] == new_data["연도"] and
            existing_data["과세표준"] == new_data["과세표준"] and
            existing_data["세율"] == new_data["세율"] and
            existing_data["누진공세"] == new_data["누진공세"]
        )

    def are_all_data_equal(self, new_data_list, existing_data_list):
        """
        새로 크롤링된 전체 데이터와 기존 전체 데이터를 비교하여 동일한지 확인.
        데이터가 동일하면 True, 그렇지 않으면 False를 반환.
        """
        if len(new_data_list) != len(existing_data_list):
            return False

        for new_record, existing_record in zip(new_data_list, existing_data_list):
            if not self.are_records_equal(existing_record, new_record):
                return False

        return True

    def save_incomeRates(self, query_year=None):

        url = 'https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2227&cntntsId=7667'
        html = self.fetch_html(url)
        
        if not html:
            return
        
        soup = BeautifulSoup(html, 'html.parser')
        
        if not soup:
            return
        
        year_table_pairs = self.extract_year_table_pairs(soup)
        
        if not year_table_pairs:
            logger.error("유효한 연도와 테이블 쌍을 찾을 수 없습니다.")
            return
        
        all_tax_data = []
        for years, table in year_table_pairs:
            table_data = self.extract_table_data(years, table)
            all_tax_data.extend(table_data)
        
        # 3. 모든 연도 데이터를 출력 (확인용)
        logger.info("모든 연도에 대한 데이터를 출력합니다:")
        json_result_all = json.dumps(all_tax_data, ensure_ascii=False, indent=4)
        logger.info(json_result_all)

        # 추가된 검증 코드: income_range 필드 검증
        for record in all_tax_data:
            if "income_range" not in record:
                logger.warning(f"'income_range' 필드가 누락되었습니다. 데이터: {record}")
                continue  # 또는 기본값을 설정할 수 있습니다


        # 기존 데이터 불러오기
        existing_data = self.vector_store.search_in_partition(query="", partition_name="IncomeRates", k=1000)

        
        # 전체 데이터를 비교하여 변경 여부 확인
        if self.are_all_data_equal(all_tax_data, existing_data):
            logger.info("기존 데이터와 새 데이터가 동일합니다. 업데이트하지 않습니다.")
        else:
            logger.info("데이터에 변경 사항이 있습니다. 전체 종합소득세 데이터를 업데이트합니다.")
            partition_name = "IncomeRates"
            self.vector_store.delete_all_data_in_partition(partition_name)
            
            try : 
                for record in all_tax_data :
                    content = {
                        "연도": record["year"],
                        "과세표준": record["income_range"],
                        "세율": record["tax_rate"],
                        "누진공세": record["deduction"]
                    }
                    
                    logger.info(f"vector_store 데이터 insert 전 partition_name : {partition_name}")
                    logger.info(f"vector_store 데이터 insert 전 content : {content}")
                    self.vector_store.add_data_to_partitions(content, partition_name=partition_name)

                logger.info(f"{record['year']}년도 종합소득세 정보가 저장되었습니다.")
                self.vector_store.search_in_partition("", partition_name, 10)
            except Exception as e :
                logger.exception(f"데이터 저장 중 오류 발생: {str(e)}")

        # 2. 특정 연도에 대한 정보를 요청하면 해당 연도의 정보를 반환하고,
        # 정보가 없는 경우 가장 최신 연도의 정보를 반환
        if query_year:
            tax_data_for_year = self.find_most_recent_year_data(all_tax_data, query_year)
        else:
            tax_data_for_year = all_tax_data
        
        # 결과를 JSON 형식으로 변환
        json_result_specific = json.dumps(tax_data_for_year, ensure_ascii=False, indent=4)
        
        # 특정 연도 데이터 출력
        logger.info(f"{query_year}년도의 데이터를 출력합니다:")
        logger.info(json_result_specific)
        

if __name__ == "__main__":
    # 현재 연도 정보 요청 (없을 경우 가장 최근 연도 반환)

    current_year = datetime.now().year
    save_income_rate = SaveIncomeRate()
    save_income_rate.save_incomeRates(query_year=current_year)
