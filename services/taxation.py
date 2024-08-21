import json
from utils.taxation_langchain import vector_store, llm, retrieval_qa
from utils.tax_utils import get_income_tax_proof, get_questions, questions_map
from services.tax_deductions import calculate_tax


class TaxationService:
    def __init__(self):
        # RetrievalQA 초기화
        self.retrieval_qa = retrieval_qa


    def answer_tax_question(self, user_input):
        """
        사용자의 세금 관련 질문에 대한 답변을 생성
        """
        response = self.retrieval_qa.run(user_input)
        return response    
    

    def getTaxation(self, taxation_data):
        """
        세무처리
        """

        taxation_result = ""

        # 1. 데이터를 VectorDB에 저장
        store_taxation_data(taxation_data)


        print("@@@@@ calculate_tax 들어가기 전 데이터 : ",taxation_data)
        # 2. 소득/세액공제 계산
        calculated_deductions = calculate_tax(taxation_data, questions_map)
        print("계산된 공제: ", calculated_deductions)  # 결과를 출력하여 확인
        taxation_result += calculated_deductions

        # 3. 거래내역 간편장부로 변환

        # 4. 종합소득세 계산

        # 5. 절세 방법 추천


        # 최종 결과 반환
        # return taxation_result
        return taxation_result



def store_taxation_data(taxation_data):
    """
    TaxationDTO 데이터를 받아 VectorDB에 저장하는 함수
    """
    # 각 데이터를 VectorDB에 저장할 때 텍스트 데이터와 결합
    def store_data(text, metadata_type):
        metadata = {
            "type": metadata_type,
            "businessId": taxation_data["businessId"]
        }
        store_with_metadata(text, metadata)
    


    # 각 데이터를 VectorDB에 저장할 떄 메타데이터 포함
    store_data(taxation_data["transactionList"]["content"], "transactionFile")
    store_data(taxation_data["incomeTaxProof"]["content"], "incomeTaxProofFile")

    print("***** transanctionList : ", taxation_data["transactionList"]["content"])
    print("***** incomeTaxList : ", taxation_data["incomeTaxProof"]["content"])
      
    # 질문과 답변을 저장
    questions = [
        f"{questions_map[1]}: {taxation_data['question1']}",
        f"{questions_map[2]}: {taxation_data['question2']}",
        f"{questions_map[3]}: {taxation_data['question3']}",
        f"{questions_map[4]}: {taxation_data['question4']}",
        f"{questions_map[5]}: {taxation_data['question5']}"
    ]

    # 답변에 따라 수정된 질문 내용 저장
    for i, question in enumerate(questions, start=1):
        if i == 4 and question.lower() == "없음":
            # 질문 4에 대한 답변이 "없음"인 경우
            question = "배우자 없음"
        store_data(question, f"question_{i}")

    store_data(taxation_data["businessId"], "businessId")
    store_data(taxation_data["businessCode"], "businessCode")
    store_data(taxation_data["currentDate"], "currentDate")
    store_data(taxation_data["bank"], "bank")
    store_data(taxation_data["businessType"], "businessType")
    store_data(taxation_data["businessContent"], "businessContent")
    store_data(taxation_data["vatInfo"], "vatInfo")
    store_data(taxation_data["incomeRates"], "incomeRates")


def store_with_metadata(text, metadata) :
    max_length = 65000
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i 
        chunk_metadata["total_chunks"] = total_chunks
        combined_text = f"METADATA: {json.dumps(chunk_metadata)}\nCONTENT: {chunk.strip()}"
        print(f"&&&&& 저장할 데이터 : {combined_text}")
        vector_store.add_texts([combined_text])

    if total_chunks > 1:
        print("데이터가 분할되어 저장되었습니다.")