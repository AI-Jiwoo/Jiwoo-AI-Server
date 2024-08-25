import logging
import json
from typing import List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from config.settings import settings
from utils.database import connect_to_milvus, get_collection
from utils.embedding_utils import get_embedding_function

logger = logging.getLogger(__name__)

class VectorStore:
    """Milvus를 사용한 벡터 저장소 클래스"""

    def __init__(self, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT):
        """
        VectorStore 초기화
        :param host: Milvus 서버 호스트
        :param port: Milvus 서버 포트
        """
        self.embedding_function = get_embedding_function()
        self.collection_name = settings.COLLECTION_NAME
        connect_to_milvus()
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """컬렉션 존재 여부 확인 및 생성"""
        if not utility.has_collection(self.collection_name):
            self._create_collection_and_index()
        else:
            self._check_and_create_index()

    def _create_collection_and_index(self):
        """컬렉션 및 인덱스 생성"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),  # URL 필드 추가
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDING_DIMENSION,
            ),
        ]
        schema = CollectionSchema(fields, "Business information for similarity search")
        collection = Collection(name=self.collection_name, schema=schema)
        self._create_index(collection)
        logger.info(f"Created collection and index: {self.collection_name}")

    def _check_and_create_index(self):
        """기존 컬렉션의 인덱스 확인 및 생성"""
        collection = get_collection(self.collection_name)
        if not collection.has_index():
            self._create_index(collection)
            logger.info(f"Created index for existing collection: {self.collection_name}")
        else:
            logger.info(f"Collection and index already exist: {self.collection_name}")

    def _create_index(self, collection):
        """인덱스 생성"""
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index("embedding", index_params)

    def add_texts(self, texts: List[str], urls: List[str] = None):
        """
        텍스트를 벡터 저장소에 추가
        :param texts: 추가할 텍스트 리스트
        :param urls: 텍스트에 해당하는 URL 리스트 (선택적)
        """
        collection = get_collection(self.collection_name)
        embeddings = [self.embedding_function(text) for text in texts]
        
        if urls is None:
            urls = [""] * len(texts)

        entities = [texts, urls, embeddings]
        collection.insert(entities)
        collection.flush()
        logger.info(f"Added {len(texts)} texts to the collection")

    def add_search_results(self, results: List[Dict[str, str]]):
        """
        검색 결과를 벡터 저장소에 추가
        :param results: 검색 결과 리스트 (각 항목은 'content'와 'url' 키를 포함하는 딕셔너리)
        """
        texts = [result['content'] for result in results]
        urls = [result['url'] for result in results]
        self.add_texts(texts, urls)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        유사도 검색 수행
        :param query: 검색 쿼리
        :param k: 반환할 결과 수
        :return: 유사한 문서 리스트
        """
        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[self.embedding_function(query)],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["content", "url"],
        )

        if not results or len(results[0]) == 0:
            logger.info("No results found in vector store.")
            return []

        hits = [
            {"content": hit.entity.get("content"), "url": hit.entity.get("url"), "metadata": {}}
            for hit in results[0] if hit.entity.get("content")
        ]

        if not hits:
            logger.info("No content found in the search results.")

        return hits

    def search_with_similarity_threshold(self, query: str, k: int = 5, threshold: float = 0.4) -> List[Dict[str, str]]:
        """
        유사도 임계값을 적용한 검색 수행
        :param query: 검색 쿼리
        :param k: 반환할 최대 결과 수
        :param threshold: 유사도 임계값
        :return: 임계값을 넘는 유사한 문서 리스트
        """
        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[self.embedding_function(query)],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["content", "url"],
        )

        if not results or len(results[0]) == 0:
            logger.info("No results found in vector store.")
            return []

        hits = []
        for hit in results[0]:
            distance = hit.distance
            similarity = 1 - (distance / max(results[0][0].distance, 1))
            if similarity >= threshold:
                hits.append({
                    "content": hit.entity.get("content"),
                    "url": hit.entity.get("url"),
                    "metadata": {"similarity": similarity}
                })

        if not hits:
            logger.info(f"No results met the similarity threshold of {threshold}.")

        return hits

    def get_relevant_info(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        주어진 쿼리에 대해 관련 정보를 검색
        :param query: 검색 쿼리
        :param k: 반환할 결과 수
        :return: 관련 정보 리스트
        """
        return self.search_with_similarity_threshold(query, k=k, threshold=settings.SIMILARITY_THRESHOLD)

    def load_documents(self, file_path: str):
        """
        파일에서 문서 로드 및 벡터 저장소에 추가
        :param file_path: 로드할 파일 경로
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        self.add_texts([doc.page_content for doc in texts])
        logger.info(f"Loaded and added documents from {file_path}")

    def _create_collection_and_partition(self):
        """
        새로운 컬렉션 생성 및 url없는 필드 정의
        파티션 생성
        """

        # 필드 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        ]

        # 컬렉션 스키마 생성
        schema = CollectionSchema(fields=fields, description="Tax Information")

        # 컬렉션 생성
        collection = Collection(self.collection_name, schema=schema)

        self._create_index(collection)

        # 파티션 생성
        collection.create_partition(partition_name="VATInfo", description="부가가치세 세율 정보")
        collection.create_partition(partition_name="IncomeRates", description="종합소득세 세율 정보")
        collection.create_partition(partition_name="TransactionFile", description="거래내역")
        collection.create_partition(partition_name="IncomeTaxFile", description="소득/세액공제 증명서류 내용")
        collection.create_partition(partition_name="Question", description="사용자 질문에 대한 응답")
        collection.create_partition(partition_name="SimpleTransaction", description="간편장부 홈페이지 내용")
        collection.create_partition(partition_name="TaxFlow", description="세액계산흐름도 홈페이지 내용")
        collection.create_partition(partition_name="TaxMethod", description="세액계산방법 홈페이지 내용")
        collection.create_partition(partition_name="TaxFiles", description="세금 관련 문서")
        collection.create_partition(partition_name="BusinessInfo", description="사업 정보")

    
    def use_custom_collection(self, custom_collection_name: str):
        """
        새로운 컬렉션을 사용하도록 설정하는 메소드.
        :param custom_collection_name : 사용할 컬렉션 이름
        """

        self.collection_name = custom_collection_name
        if not utility.has_collection(self.collection_name):
            self._create_collection_and_partition()
        else :
            self._check_and_create_index()
    
    
    def add_texts_with_metadata(self, texts: List[str], metadatas: List[Dict]):
        """
        텍스트와 메타데이터를 함께 벡터 저장소에 추가
        :param texts: 추가할 텍스트 리스트
        :param metadatas: 텍스트에 해당하는 메타데이터 리스트
        """

        collection = get_collection(self.collection_name)
        embeddings = [self.embedding_function(text) for text in texts]

        entities = [texts, metadatas,embeddings]
        collection.insert(entities)
        collection.flush()
        logger.info(f"컬렉션에 metadata와 함꼐 {len(texts)}개의 텍스트를 저장했습니다.")

    def add_data_to_partitions(self, texts: Dict, partition_name: str):
        """
        특정 파티션에 데이터를 추가
        : param texts : 추가할 텍스트 리스트
        : param partition_name : 데이터를 추가할 파티션 이름
        """
        logger.info(f"add_data_to_partitions 진입 시 partition_name: {partition_name}")
        collection = get_collection(self.collection_name)

        # texts 리스트의 구조를 로깅하여 확인
        # logger.debug(f"Texts provided: {texts}")

        # texts 리스트에서 필요한 텍스트 필드 추출
        text_json = json.dumps(texts, ensure_ascii=False)  # 딕셔너리 전체를 JSON 문자열로 변환하여 저장
        
        # 텍스트 데이터를 임베딩 벡터로 변환
        embeddings = self.embedding_function(text_json) 

        data_to_insert = {
            "content": text_json,
            "embedding": embeddings,
        }

        # 파티션 지정하여 데이터 삽입
        collection.insert(data=data_to_insert, partition_name=partition_name)
        collection.flush()
        logger.info(f"{len(texts)}개의 텍스트를 {partition_name} 파티션에 추가하였습니다.")


    def search_in_partition(self, query: str, partition_name: str, k: int = 5)-> List[Dict[str, str]]:
        """
        특정 파티션에서 유사도 검색 수행
        :param query : 검색 쿼리
        :param partition_name: 검색할 파티션 이름
        :param k : 반환할 결과 수
        :return : 유사한 문서 리스트
        """

        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[self.embedding_function(query)],
            anns_field="embedding",
            param=search_params,
            limit=k,
            partition_names=[partition_name],
            output_fields=["content"]
        )

        if not results or len(results[0]) == 0 :
            logger.info(f"{partition_name}의 이름의 파티션을 찾을 수 없습니다.")
            return []

        hits = [{"content": hit.entity.get("content")} for hit in results[0] if hit.entity.get("content")]

        if not hits : 
            logger.info(f"{partition_name} 파티션에서 검색 결과를 찾을 수 없습니다.")
        
        return hits
    
    def search_in_all_partitions(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        모든 파티션에서 유사도 검색 수행
        :param query: 검색 쿼리
        :param k: 각 파티션별로 반환할 결과 수
        :return: 모든 파티션에서의 유사한 문서 리스트
        """
        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # 모든 파티션 이름 가져오기
        partitions = collection.partitions

        all_results = []

        for partition in partitions:
            results = collection.search(
                data=[self.embedding_function(query)],
                anns_field="embedding",
                param=search_params,
                limit=k,
                partition_names=[partition.name],
                output_fields=["content"]
            )

            if results and len(results[0]) > 0:
                hits = [
                    {
                        "content": hit.entity.get("content"),
                        "partition": partition.name,
                        "similarity": 1 - (hit.distance / max(results[0][0].distance, 1))
                    }
                    for hit in results[0] if hit.entity.get("content")
                ]
                all_results.extend(hits)

        # 전체 결과를 유사도 기준으로 정렬
        all_results.sort(key=lambda x: x['similarity'], reverse=True)

        if not all_results:
            logger.info("No results found in any partition.")

        return all_results[:k]  # 전체 중 상위 k개 결과만 반환


    def delete_all_data_in_partition(self, partition_name : str):
        """
        특정 파티션 내의 모든 데이터를 삭제
        :param partition_name: 데이터를 삭제할 파티션 이름
        """
        collection = get_collection(self.collection_name)
        
        # 특정 파티션에서 모든 데이터를 삭제하는 expr 조건
        expr = "id >= 0"  # id가 0 이상인 모든 데이터 삭제
        
        collection.delete(expr=expr, partition_name=partition_name)
        collection.flush()
        logger.info(f"{partition_name} 파티션에서 모든 데이터를 삭제하였습니다.")
            
    def delete_all_indexs(self):
        """
        한 컬렉션의 모든 인덱스 삭제
        """
        collection = get_collection(self.collection_name)
        collection.drop_index()
        logger.info(f"Collection '{self.collection_name}'의 모든 인덱스가 삭제되었습니다.")
