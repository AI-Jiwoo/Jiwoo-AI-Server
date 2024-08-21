from services.taxation import vector_store
from utils.vector_store_retriever import VectorStoreRetriever

retriever = VectorStoreRetriever(vector_store=vector_store)

questions_map = {
    1: "현재 부양하고 있는 가족(배우자, 자녀, 부모 등)의 수",
    2: "부양가족 중 연간 소득이 100만 원을 초과하지 않는 가족의 수",
    3: "부양하는 각 자녀의 나이",
    4: "배우자의 연간소득이 100만원을 초과하는지 여부 (배우자가 없다면 없음이라고 적어주세요)",
    5: "부양가족 중 장애인으로 등록된 사람 수"
}

def get_income_tax_proof(business_id):
    """
    businessId 로 사용자의 소득/세액공제 데이터 가져오기
    """
    print(f"★★★★★★ get_income_tax_proof 들어옴 businessId : {business_id}" )
    query = f'businessId: "{business_id}" AND type: "incomeTaxProofFile"'
    # results = vector_store.similarity_search(query=query, k=10)

    # 검색 실행
    results = retriever(query)
    print(f"검색된 결과: {results}")
    # income_tax_proof = vector_store.similarity_search(query=query, k=1)
    # for result in results:
    #     if 'incomeTaxProofFile' in result['content']:
            
    #         content = result['content']
    #         print(f"★★★★★★ income_tax_proof 있대요 : {content}")
    #         return content

    if results:
        for result in results:
            content = result['content']
            metadata = result['metadata']
            print(f"★★★★★★ income_tax_proof 있대요 : {content}, 메타데이터 : {metadata}")
            return content
        
    print("☆☆☆☆☆☆사용자의 소득/세액공제 불러오기 실패")
    return None

    
    # if income_tax_proof:
    #     content = income_tax_proof[0]['content']
    #     print(f"★★★★★★ content : {content}")
    #     content = content.split("CONTENT:")[1].strip() if "CONTENT:" in content else content.strip()
    #     print(f"★★★★★★ income_tax_proof 있대요 : {content}")
    #     return content
    # else :
    #     print("☆☆☆☆☆☆사용자의 소득/세액공제 불러오기 실패")
    # return None

def get_questions(business_id):
    """
    사용자의 질문의 답변 5가지 가져오기
    """
    questions = []
    for i in range(1, 6):  # 질문 1~5
        query = f'businessId:"{business_id}" AND type: "question_{i}"'
        # result = vector_store.similarity_search(query=query, k=1)
        results = retriever(query)
        if results:
            questions.append(results[0]['content'])
        else : 
            print("☆☆☆☆☆ 사용자의 질문 답변 가져오기 실패")
    return questions


def get_tax_documents():
    """
    세액 관련 자료 가져오기
    """
    query = "type:tax_document"
    tax_documents = vector_store.similarity_search(query=query, k=10)
    return [doc['content'] for doc in tax_documents] if tax_documents else []
     