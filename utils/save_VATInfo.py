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

class SaveVATInfo : 

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

    def parse_html(self, html):
        return BeautifulSoup(html, 'html.parser')

    def extract_date_from_caption(self, caption):
        pattern = re.compile(r'\d{4}\.\d{1,2}\.\d{1,2}')
        match = pattern.search(caption)
        if match:
            date_str = match.group()
            return datetime.strptime(date_str, "%Y.%m.%d").date()
        logger.warning("기준일을 찾을 수 없습니다.")
        return None

    def extract_tables_with_dates(self, soup):
        table_data = []
        h3_tags = soup.find_all('h3', class_='tit1')

        for h3 in h3_tags:
            h3_text = h3.text.strip()
            date = self.extract_date_from_caption(h3_text)
            table = h3.find_next('table')
            
            if table:
                if "일반과세자" in h3.text:
                    tax_type = "일반과세자"
                    period = "no_date" if not date else None
                elif "간이과세자" in h3.text:
                    tax_type = "간이과세자"
                    if "이전" in h3.text:
                        period = "before"
                    elif "이후" in h3.text:
                        period = "after"
                else:
                    tax_type = "알 수 없음"
                    period = None

                table_data.append((tax_type, period, date, table))
            else:
                logger.warning(f"{h3.text.strip()}에 해당하는 테이블을 찾을 수 없습니다.")
        
        return table_data

    def extract_table_data(self, tax_type, period, date, table):
        table_data = []
        rows = table.find_all('tr')

        for row in rows[1:]:  # 첫 번째 행은 헤더로 제외
            columns = row.find_all('td')
            if len(columns) >= 2:
                category = columns[0].text.strip()
                tax_rate = columns[1].text.strip()
                
                entry = {
                    "tax_type": tax_type,
                    "period": period if period else "no_date",
                    "date": str(date) if date else "unknown",
                    "category": category,
                    "tax_rate": tax_rate
                }
                table_data.append(entry)
            else:
                logger.warning(f"잘못된 형식의 행을 발견했습니다: {row}")
        
        return table_data

    def get_existing_vat_data(self):
        """
        기존의 부가가치세 데이터 가져오기
        """
        return self.vector_store.search_in_partition(query="", partition_name="VATInfo", k=1000)

    def are_records_equal(self, existing_record, new_record):
        """
        기존 레코드와 새 레코드를 비교하여 동일한지 확인합니다.
        동일하면 True, 그렇지 않으면 False를 반환합니다.
        """
        # 기존 레코드의 content는 JSON 문자열로 저장되어 있으므로, 이를 파싱하여 비교합니다.
        existing_data = json.loads(existing_record['content'])

        # 새 레코드도 동일한 구조로 변환합니다.
        new_data = {
            "사업자 유형": new_record["tax_type"],
            "시기": new_record["period"],
            "기준 날짜": new_record["date"],
            "업종": new_record["category"],
            "부가가치율": new_record["tax_rate"],
            "부가가치세 세율": new_record["tax_rate"],
            "세율": new_record["tax_rate"]
        }

        # 모든 필드를 비교합니다.
        return (
            existing_data["사업자 유형"] == new_data["사업자 유형"] and
            existing_data["시기"] == new_data["시기"] and
            existing_data["기준 날짜"] == new_data["기준 날짜"] and
            existing_data["업종"] == new_data["업종"] and
            existing_data["부가가치율"] == new_data["부가가치율"] and
            existing_data["부가가치세 세율"] == new_data["부가가치세 세율"] and
            existing_data["세율"] == new_data["세율"]
        )

    def are_all_data_equal(self, new_data_list, existing_data_list):
        """
        새로 크롤링된 전체 데이터와 기존 전체 데이터를 비교하여 동일한지 확인.
        데이터가 동일하면 True, 그렇지 않으면 False를 반환
        """
        if len(new_data_list) != len(existing_data_list):
            return False
        
         # 기존 데이터와 새 데이터를 정렬한 후 비교 (정렬 기준을 원하는 필드로 설정 가능)
        sorted_new_data_list = sorted(new_data_list, key=lambda x: (x['tax_type'], x['period'], x['date'], x['category']))
        sorted_existing_data_list = sorted(existing_data_list, key=lambda x: (json.loads(x['content'])['사업자 유형'], json.loads(x['content'])['시기'], json.loads(x['content'])['기준 날짜'], json.loads(x['content'])['업종']))

        
        for new_record, existing_record in zip(sorted_new_data_list, sorted_existing_data_list):
            if not self.are_records_equal(existing_record, new_record):
                return False
        
        return True

    def get_relevant_data(self, all_tax_data, query_date, cutoff_data):

        relevant_data = []
        
        for tax_type in ["일반과세자", "간이과세자"]:
            tax_data = [data for data in all_tax_data if data["tax_type"] == tax_type]
            if tax_data:
                before_cutoff = [data for data in tax_data if data["period"] == "before"]
                after_cutoff = [data for data in tax_data if data["period"] == "after"]
                no_date_data = [data for data in tax_data if data["period"] == "no_date"]

                if before_cutoff or after_cutoff:
                    # 쿼리 날짜와 기준일 비교
                    if query_date <= cutoff_data:
                        relevant_data.extend(before_cutoff)
                    else:
                        relevant_data.extend(after_cutoff)
                else:
                    # 일반과세자에 날짜가 없는 경우 또는 간이과세자의 기준일 이전 데이터가 없는 경우
                    relevant_data.extend(no_date_data)
        
        return relevant_data

    def save_VAT_info(self, query_date=None):
        url = 'https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2275&cntntsId=7696'
        html = self.fetch_html(url)
        
        if not html:
            return
        
        soup = self.parse_html(html)
        
        if not soup:
            return
        
        tables_with_dates = self.extract_tables_with_dates(soup)
        
        if not tables_with_dates:
            logger.error("유효한 날짜와 테이블 쌍을 찾을 수 없습니다.")
            return
        
        all_tax_data = []
        for tax_type, period, date, table in tables_with_dates:
            table_data = self.extract_table_data(tax_type, period, date, table)
            all_tax_data.extend(table_data)
        
        # 3. 모든 데이터를 출력 (확인용)
        logger.info("모든 데이터를 출력합니다:")
        json_result_all = json.dumps(all_tax_data, ensure_ascii=False, indent=4)
        print(json_result_all)

        
        # 기존 데이터 불러오기
        existing_data = self.get_existing_vat_data()

        # 전체 데이터를 비교하여 변경 여부 확인
        if self.are_all_data_equal(all_tax_data, existing_data) :
            logger.info("기존 데이터와 새 데이터가 동일합니다. 업데이트하지 않습니다.")
        else:
            logger.info("데이터에 변경사항이 있습니다. 전체 부가가치세 데이터를 업데이트합니다.")
            partition_name = "VATInfo"
            self.vector_store.delete_all_data_in_partition(partition_name)


            try:
                for record in all_tax_data:
                    content = {
                        "사업자 유형": record["tax_type"],
                        "시기": record["period"],
                        "기준 날짜": record["date"],
                        "업종": record["category"],
                        "부가가치율": record["tax_rate"],
                        "부가가치세 세율": record["tax_rate"],
                        "세율": record["tax_rate"]
                    }

                    logger.info(f"vector_store 데이터 insert 전 partition_name : {partition_name}")
                    logger.info(f"vector_store 데이터 insert 전 content : {content}")
                    self.vector_store.add_data_to_partitions(content, partition_name=partition_name)
                
                if all_tax_data:
                    logger.info(f"{all_tax_data[-1]['date']} 부가가치세 정보가 저장되었습니다.")

                logger.info(f"{record['date']} 부가가치세 정보가 저장되었습니다.")
                self.vector_store.search_in_partition("", partition_name, 10)
            
            
            
            except Exception as e:
                logger.exception(f"데이터 저장 중 오류 발생: {str(e)}")


        # 특정 날짜에 대한 정보를 요청하면 해당 날짜의 정보를 반환하고,
        # 정보가 없는 경우 가장 최근 날짜의 정보를 반환
        if query_date:
            tax_data_for_date = self.get_relevant_data(all_tax_data, query_date, max([d for t, p, d, _ in tables_with_dates if d]))
        else:
            tax_data_for_date = all_tax_data
        
        # 결과를 JSON 형식으로 변환
        json_result_specific = json.dumps(tax_data_for_date, ensure_ascii=False, indent=4)
        
        # 특정 날짜 데이터 출력
        logger.info(f"{query_date}의 데이터를 출력합니다:")
        print(json_result_specific)
        
        # 필요에 따라 결과를 파일로 저장
        # with open('vat_data.json', 'w', encoding='utf-8') as f:
        #     f.write(json_result_specific)

if __name__ == "__main__":
    # 현재 날짜 정보 요청 (없을 경우 가장 최근 날짜 반환)
    query_date = datetime(2024, 1, 1).date()
    save_VATinfo = SaveVATInfo()

    save_VATinfo.save_VAT_info(query_date=query_date)
