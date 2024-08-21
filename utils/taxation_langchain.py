import logging
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from utils.vector_store import VectorStore
from utils.vector_store_retriever import VectorStoreRetriever

# 로깅 설정
logger = logging.getLogger(__name__)

# LLM 모델 초기화 (GPT-4 사용)
llm = ChatOpenAI(model="gpt-4")

# VectorStore 초기화
vector_store = VectorStore()

# VectorStoreRetriever 초기화
retriever = VectorStoreRetriever(vector_store)


# RAG RetrievalQA 구성
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
