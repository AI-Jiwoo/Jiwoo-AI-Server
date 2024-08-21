from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
import requests
import json

from utils.tax_calculation_url import store_tax_calculation_data
from utils.tax_utils import get_income_tax_proof, get_questions, get_tax_documents, questions_map
from utils.taxation_langchain import vector_store, llm, retrieval_qa
from utils.web_search import WebSearch


def load_tax_calculation_data():
    """
    세액 계산흐름도와 세액계산방법 데이터를 VectorDB에 저장
    """
    store_tax_calculation_data()




def calculate_tax(taxation_data, questions_map):
    """
    소득/세액 공제 계산
    """
    
    business_id = taxation_data["businessId"]
    print(f"##### business_id : {business_id}")
 
    # 사용자의 소득/세액공제 파일 가져오기
    user_documents = get_income_tax_proof(business_id)
    print(f"##### user_documents : {user_documents}")

    # vectorDB에 세액 계산법과 세액 계산 흐름도 저장 및 업데이트
    load_tax_calculation_data()
    
    tax_calculation_flow = get_tax_calculation_flow()
    tax_calculation_method = get_tax_calculation_method()
    print(f"##### tax_calculation_flow: {tax_calculation_flow}")  # 디버깅 출력
    print(f"##### tax_calculation_method: {tax_calculation_method}")  # 디버깅 출력


    # 사용자의 답변 가져오기
    questions = get_questions(business_id)
    questions_with_context = "\n".join([f"{questions_map[i+1]}: {q}" for i, q in  enumerate(questions)])  # 문자열로 변환
    print(f"##### questions_with_context: {questions_with_context}")  # 디버깅 출력


    # 세액 관련 자료 가져오기
    tax_documents = get_tax_documents()
    print(f"##### tax_documents: {tax_documents}")

    # 데이터 확인
    print("***** 질문 답변 : ", questions_with_context)
    print("***** 세액 계산 방법 : ", tax_calculation_method)
    print("***** 세액 계산 흐름도 : ", tax_calculation_flow)
    print("***** 세금 관련 서류 ", tax_documents)

    # 총 소득공제/총 세액공제 계산하는 법을 url을 참고하여 gpt로 추출 및 총 소득공제/총 세액공제 계산 
    context_documents = [user_documents, questions_with_context, tax_calculation_method, tax_calculation_flow] + tax_documents
    context = "\n".join(context_documents)
    print(f"##### 모든 정보 : {context}") 

    prompt_template = """
    당신은 세무 전문가입니다. 아래 문서와 이용자의 질문에 대한 답변을 바탕으로 관련된 총 소득공제와 총 세액공제를 계산하는 식을 구하고, 해당 식으로 총 소득공제와 총 세액공제를 구해주세요.
    세액 계산 방법과 세액 계산 흐름도와 세액 관련 서류를 참고하여 계산해주세요.
    단, 답변은 정해진 형식을 지켜주세요. 계산식에 들어가는 항목은 '의료비 인별합계금액' 과 같이 문자로 표현해주세요.
    금액이 '만 원'을 넘어가면 단위를 '원'에서 '만 원'으로 바꿔주세요.
    그리고 계산식에 있는 항목들이 정확한지, 정보가 확실한지 여부도 명시해주세요.

    문서: {user_documents}
    질문: {questions_with_context}
    세액 계산 방법: {tax_calculation_method}
    세액 계산 흐름도: {tax_calculation_flow}
    세액 관련 서류: {tax_documents}

    답변 형식:
    총 소득공제 계산식: [구체적인 계산식 예시]
    총 세액공제 계산식: [구체적인 계산식 예시]
    총 소득공제: [정확한 금액] 원
    총 세액공제: [정확한 금액] 원

    가능한 경우, 각각의 정보를 기반으로 구체적인 계산식을 제공하세요.
    답변에 모든 정보가 포함되어 있는지 확인하고, 누락된 정보나 불확실한 부분이 있으면 명확히 지적하십시오.

    """

    prompt = PromptTemplate(
        input_variables=["user_documents", "questions_with_context", "tax_calculation_method", "tax_calculation_flow", "tax_documents"],
        template=prompt_template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(
        user_documents=user_documents,
        questions_with_context=questions_with_context,
        tax_calculation_method=tax_calculation_method,
        tax_calculation_flow=tax_calculation_flow,
        tax_documents="\n".join(tax_documents)  # tax_documents도 문자열로 변환

        )


     # 확실하지 않은 정보가 있는지 확인
    if "확실하지 않은 정보" in result:
        uncertain_items = extract_uncertain_items(result)
        
        for item in uncertain_items:
            search_result = search_for_tax_info(item, questions, user_documents)
            result += search_result

    return result


def extract_uncertain_items(result):
    """
    GPT-4 결과에서 확실하지 않은 항목을 추출
    """
    # 예시로 단순히 텍스트 파싱을 통해 "확실하지 않은 정보" 항목을 추출
    start_keyword = "확실하지 않은 정보 :"
    uncertain_section = result.split(start_keyword)[1].strip()
    uncertain_items = uncertain_section.split(",")  # 항목들을 ','로 구분했다고 가정
    return [item.strip() for item in uncertain_items if item]


# 웹 검색
def search_for_tax_info(topic, questions, user_documents):
    """
    주어진 주제(소득공제, 세액공제 등)에 대해 웹에서 정보를 검색하고 GPT로 다시 계산
    """
    # search_query = f"{topic} 관련 최신 정보"
    # retriever = vector_store.as_retriever(search_type="similarity")
    
    # 웹 검색을 통해 관련 자료 찾기
    # search_results = retriever.get_relevant_documents(search_query)
    # search_results = perform_web_search(search_query)
    web_search = WebSearch()
    search_results = web_search.search(topic, num_results=3)
    
    # 검색된 정보를 바탕으로 GPT에게 추가 계산 요청
    prompt_template = """
    아래는 {topic}에 대한 검색 결과입니다. 이 정보를 바탕으로, 이용자의 질문에 대한 답변과 소득/세액공제 파일 내용을 참고하여 정확한 계산을 다시 수행해주세요.

    검색 결과 : {search_results}
    문서 : {user_documents}
    질문 : {questions}
    
    답변 형식 : 
    총 소득공제 계산식 : a + b + c + ...
    총 세액공제 계산식 : a + b + c + ...
    총 소득공제 : 0 원 
    총 세액공제 : 0 원
    """

    prompt = PromptTemplate(
        input_variables=["topic", "search_results", "user_documents", "questions"],
        template=prompt_template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(
        topic=topic,
        search_results="\n".join([result['content'] for result in search_results]),  # 리스트를 문자열로 변환
        user_documents=user_documents,
        questions=questions
    )

    return result


def get_tax_calculation_flow():
    """
    세액계산흐름도 가져오기
    """

    query = "type:tax_calculation_flow"
    tax_calculoation_flow = vector_store.similarity_search(query=query, k=1)
    if tax_calculoation_flow:
         return tax_calculoation_flow[0]['content']
    return None


def get_tax_calculation_method():
    """
    세액계산방법 가져오기
    """
    query = "type:tax_calculation_method"
    tax_calculation_method = vector_store.similarity_search(query=query, k=1)
    
    if tax_calculation_method:
        # 데이터를 JSON 형태로 저장했으므로, 이를 다시 파싱
        result = json.loads(tax_calculation_method[0]['content'])
        return result['content']  # 실제 내용만 반환
    return None


