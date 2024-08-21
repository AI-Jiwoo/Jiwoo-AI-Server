from langchain.schema import BaseRetriever
from typing import Any
import json

class VectorStoreRetriever(BaseRetriever):
    vector_store: Any # 'Any'를 사용하여 vector_store의 타입을 지정

    def __init__(self, vector_store: Any):
        super().__init__()
        self.vector_store = vector_store

    def _get_relevant_documents(self, query, k=5):
        # VectorStore의 similarity_search 결과를 적절한 형식으로 반환
        result = self.vector_store.similarity_search(query, k=k)
    

        return result


    def __call__(self, query):
        # `retriever`가 호출될 때 `get_relevant_documents` 메서드를 통해 결과를 반환
        return self._get_relevant_documents(query)

    def _parse_metadata_and_content(self, results):
        """
        검색된 결과에서 메타데이터와 본문을 분리하여 반환
        :param results: 검색된 결과 리스트
        :return: 메타데이터와 본문이 분리된 리스트
        """
        parsed_results = []
        
        for result in results:
            content = result['content']
            
            # 메타데이터와 본문을 분리
            if content.startswith("METADATA:"):
                try:
                    metadata_part, content_part = content.split("\nCONTENT:", 1)
                    metadata = json.loads(metadata_part[len("METADATA: "):])
                    content = content_part.strip()
                    parsed_results.append({"metadata": metadata, "content": content})
                except ValueError:
                    # 파싱에 실패한 경우 원본을 유지
                    parsed_results.append({"metadata": None, "content": content})
            else:
                parsed_results.append({"metadata": None, "content": content})
        
        return parsed_results