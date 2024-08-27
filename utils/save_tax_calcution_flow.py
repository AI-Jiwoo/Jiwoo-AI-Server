import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from utils.vector_store import VectorStore
from datetime import datetime

from config.settings import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaveTaxCalculationFlow : 
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)


    def get_existing_data(self, partition_name):
        """
        기존의 데이터를 조회
        """
        return self.vector_store.search_in_partition(query="", partition_name=partition_name, k=1000)

    def are_data_equal(self, new_data, existing_data):
        """
        새로 크롤링한 데이터와 기존 데이터를 비교하여 동일한지 확인
        데이터가 동일하면 True, 그렇지 않으면 False를 반환
        """
        return new_data == existing_data

    def save_tax_flow_to_vectordb(self, tax_flow_dict):
        try:
             
            partition_name = "TaxFlow"
            
            content = {
                "url": tax_flow_dict.get("url", "unknown"),
                "세액흐름도 내용": tax_flow_dict.get("세액흐름도 내용", {})
            }

            # 기존 데이터 가져오기
            existing_data = self.get_existing_data(partition_name)

            # 데이터 비교 및 업데이트 로직
            if existing_data : 
                if self.are_data_equal(content, existing_data):
                    logger.info("기존 데이터와 새 데이터가 동일합니다.")
                else:
                    logger.info("데이터에 변경 사항이 있습니다. 업데이트를 진행합니다.")
                    self.vector_store.delete_all_data_in_partition(partition_name)
                    self.vector_store.add_data_to_partitions(content,  partition_name=partition_name)
                    logger.info("데이터가 업데이트되었습니다.")
            else:
                logger.info("기존 데이터가 없습니다. 새 데이터를 저장합니다.")
                self.vector_store.add_data_to_partitions(content, partition_name=partition_name)
                logger.info("새 데이터가 저장되었습니다.")
        
        except Exception as e:
            logger.exception(f"데이터 저장 중 오류 발생: {str(e)}")


    def save_tax_flow(self):
        # GPT가 생성한 dict 구조

        # GPT에게 물어보는 프롬프트
            # 이미지 분석 및 데이터 구조화 요청:
            # 다음 이미지는 "종합소득세 계산 과정"을 도식화한 것입니다. 이 이미지를 보고, 이미지에 포함된 모든 정보를 분석하여 설명과 함께 dict 형식으로 변환해 주세요. 첫 줄에 불필요한 설명은 없애고, 오직 dict 구조만 출력해 주세요. 특히 "기납부세액"에 포함되는 항목들이 반드시 포함되도록 해 주세요.
            # 요청 사항:
            # 이미지를 철저히 분석하여 필요한 모든 항목을 추출해 주세요.
            # 각 항목에 대한 설명을 추가하고, 이를 dict 구조로 변환해 주세요. GPT가 직접 구조를 생성해야 합니다.
            # "기납부세액"의 세부 항목 등 모든 중요한 정보가 누락되지 않도록 주의해 주세요. 
            # 최종 dict 구조만을 출력해 주세요. 추가 설명이나 불필요한 문장은 포함하지 말아 주세요.

        tax_flow_dict = {
            "content": {
                "url": "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2226&cntntsId=7666",  # 세액흐름도 이미지나 페이지의 URL
                "세액흐름도 내용": {
                        "소득 종류": {
                            "설명": "종합소득세를 계산하기 위해 합산되는 소득의 종류",
                            "내용": [
                                "이자소득: 예금, 채권 등으로부터 발생하는 이자 수익",
                                "배당소득: 주식, 펀드 등의 배당금 수익",
                                "사업소득(부동산 임대): 사업 활동과 부동산 임대를 통해 얻는 소득",
                                "근로소득: 고용관계에서 발생하는 급여, 보너스 등의 소득",
                                "연금소득: 연금으로부터 발생하는 소득",
                                "기타소득: 상금, 복권 당첨금, 로열티 등 일시적 소득"
                            ]
                        },
                        "종합소득금액": {
                            "설명": "다양한 소득을 합산한 총 소득 금액으로, 과세의 기초가 되는 금액",
                            "내용": "이자소득, 배당소득, 사업소득, 근로소득, 연금소득, 기타소득을 합산하여 도출"
                        },
                        "소득공제": {
                            "설명": "종합소득금액에서 공제할 수 있는 항목들로, 과세표준을 낮추는 역할",
                            "내용": {
                                "기본공제": "본인, 배우자, 부양가족에 대한 공제",
                                "추가공제": "경로우대(고령자), 장애인 등에 대한 추가 공제",
                                "연금보험료공제": "연금보험료 납부에 따른 공제",
                                "주택담보노후연금 이자비용공제": "주택담보대출 이자비용에 대한 공제",
                                "특별소득공제": "본적률 공제, 주택자금 공제 등 주거 관련 공제",
                                "조특법상 소득공제": "주택마련저축, 신용카드 사용금액, 소기업·소상공인 공제부금, 장기집합투자증권저축 등"
                            }
                        },
                        "종합소득 과세표준": {
                            "설명": "소득공제를 적용한 후 남은 금액으로, 실제로 세금을 부과할 기준 금액",
                            "내용": "종합소득금액에서 소득공제를 차감하여 계산"
                        },
                        "세율": {
                            "설명": "종합소득 과세표준에 따라 적용되는 세율로, 누진세율 구조",
                            "내용": "과세표준에 따라 6% ~ 45%의 세율 적용"
                        },
                        "산출세액": {
                            "설명": "과세표준에 세율을 적용하여 계산된 기본 세액",
                            "내용": "종합소득 과세표준에 세율을 적용하여 산출"
                        },
                        "세액공제 및 세액감면": {
                            "설명": "산출세액에서 추가적으로 공제할 수 있는 항목들로, 최종 세액을 줄이는 역할",
                            "내용": {
                                "특별세액공제": "본적률, 의료비, 교육비, 기부금, 표준세액공제 등",
                                "기장세액공제": "장부 기장을 통해 세액을 공제받는 경우",
                                "재해손실세액공제": "재해로 인한 손실에 대한 세액 공제",
                                "근로소득세액공제": "근로소득에 대한 세액공제",
                                "배당세액공제": "배당소득에 대한 세액공제",
                                "외국납부세액공제": "외국에서 이미 납부한 세금에 대한 공제",
                                "농어업인에 대한 세액감면": "농업 또는 어업에 종사하는 사람들에게 제공되는 세액감면",
                                "전자신고세액공제": "전자 방식으로 세금을 신고한 경우 제공되는 공제",
                                "성실신고사업자에 대한 세액감면": "성실하게 세금을 신고한 사업자에게 제공되는 감면",
                                "중소기업특별세액감면": "중소기업에 제공되는 특별 세액감면"
                            }
                        },
                        "가산세": {
                            "설명": "세액 산출 후에 추가적으로 부과될 수 있는 세금 항목",
                            "내용": {
                                "무신고가산세": "세금을 신고하지 않은 경우 부과",
                                "과소신고가산세 및 초과환급신고가산세": "세금을 적게 신고하거나 초과 환급을 받은 경우 부과",
                                "납부지연가산세": "세금을 기한 내에 납부하지 않은 경우 부과",
                                "중납불비가산세": "중간 납부 불이행 시 부과",
                                "무기장가산세": "장부 기장을 하지 않은 경우 부과"
                            }
                        },
                        "기납부세액": {
                            "설명": "이미 납부한 세액이 있는 경우 이를 차감하여 계산",
                            "내용": {
                                "중간예납세액": "중간에 미리 납부한 세액",
                                "수시부과세액": "수시로 부과된 세액",
                                "원천징수세액": "원천징수로 납부된 세액"
                            }
                        },
                        "납부할 세액": {
                            "설명": "모든 공제, 감면, 가산세 등을 반영한 후 최종적으로 납부해야 할 세액",
                            "내용": "기납부세액을 차감한 후의 최종 금액"
                        }
                }
            }
        }

        # 위 dict를 VectorDB에 저장하는 함수 호출
        self.save_tax_flow_to_vectordb(tax_flow_dict["content"])


if __name__ == "__main__":
    save_tax_calculation_flow = SaveTaxCalculationFlow()

    save_tax_calculation_flow.save_tax_flow()
