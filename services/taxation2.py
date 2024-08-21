from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
from config.settings import settings

def Tax_service(taxationDTO): 
    # taxationDTO는 이미 딕셔너리로 들어오므로 바로 사용
    wrapped_data = {"user_data": taxationDTO}

    # 결과를 JSON으로 변환하여 출력 (테스트용)
    wrapped_json = json.dumps(wrapped_data, indent=4)
    print(wrapped_json)

    # transactionList와 incomeTaxProof의 content에서 특수 문자를 이스케이프 처리
    escaped_tranList = taxationDTO["transactionList"]["content"].replace('\n', '\\n').replace('"', '\\"')
    escaped_taxProof = taxationDTO["incomeTaxProof"]["content"].replace('\n', '\\n').replace('"', '\\"')

    # 이스케이프된 내용을 다시 딕셔너리에 할당
    taxationDTO["transactionList"]["content"] = escaped_tranList
    taxationDTO["incomeTaxProof"]["content"] = escaped_taxProof

    # 페이지 콘텐츠 = transactionList + incomeTaxProof
    page_content = (
        f"Transactions:\n{taxationDTO['transactionList']['content']}\n\n"
        f"Income Tax Proof:\n{taxationDTO['incomeTaxProof']['content']}"
    )
    print("page_content : " + page_content)

    # 메타데이터 생성
    metadata = {
        "transactionSource": taxationDTO['transactionList']['fileName'],
        "incomeTaxProofSource": taxationDTO['incomeTaxProof']['fileName'],
        "businessId": taxationDTO['businessId'],
        "currentDate": taxationDTO['currentDate'],
        "bank": taxationDTO['bank'],
        "businessType": taxationDTO['businessType'],
        "businessContent": taxationDTO['businessContent'],
        "vatInfo": taxationDTO['vatInfo'],
        "incomeRates": taxationDTO['incomeRates']
    }

    # Document 객체로 통합
    combined_document = Document(
        page_content=page_content,
        metadata=metadata
    )


    print("Combined Document Metadata:", combined_document.metadata)
    print("\nCombined Document Page Content Preview:\n", combined_document.page_content[:500])

    # Split data small chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 400, 
        chunk_overlap=20, 
        length_function=len, 
        add_start_index = True
    )
    all_splits = text_splitter.split_documents([combined_document])

    print(all_splits)

    URI = os.getenv("MILVUS_HOST", settings.MILVUS_HOST) + os.getenv("MILVUS_PORT", settings.MILVUS_PORT)

    

# 테스트 호출
imsi_data = {
    "transactionList": 
    {
        "fileName": "transaction_list.txt",
        "content": "transaction content here"
    },
    "incomeTaxProof": 
    {
        "fileName": "income_tax_proof.txt",
        "content": "income tax proof content here"
    },
    "question1": "답변1",
    "question2": "답변2",
    "question3": "답변3",
    "question4": "답변4",
    "question5": "답변5",
    "businessId": "123456",
    "businessCode": "654321",
    "currentDate": "2024-08-21",
    "bank": "Some Bank",
    "businessType": "Some Business Type",
    "businessContent": "Business content here",
    "vatInfo": "VAT info here",
    "incomeRates": "Income rates here"
}

Tax_service(imsi_data)
