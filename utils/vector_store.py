import logging

import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from typing import List, Dict, Any
from datetime import datetime
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from config.settings import settings
from utils.database import connect_to_milvus, get_collection
from utils.embedding_utils import get_embedding_function
from services.models import CompanyInfo, SupportProgramInfo

logger = logging.getLogger(__name__)

class VectorStore:
    """Milvus를 사용한 벡터 저장소 클래스"""

    def __init__(self, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT):
        # 벡터 저장소 초기화
        self.embedding_function = get_embedding_function()
        self.collection_name = settings.COLLECTION_NAME
        connect_to_milvus()
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        # 컬렉션 존재 여부 확인 및 생성
        if not utility.has_collection(self.collection_name):
            self._create_collection_and_index()
        else:
            self._check_and_create_index()

    def _create_collection_and_index(self):
        # 컬렉션 및 인덱스 생성
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION),
            FieldSchema(name="created_at", dtype=DataType.INT64), 
        ]
        schema = CollectionSchema(fields, "비즈니스 정보 유사도 검색을 위한 스키마")
        collection = Collection(name=self.collection_name, schema=schema)
        self._create_index(collection)
        logger.info(f"컬렉션 및 인덱스 생성 완료: {self.collection_name}")

    def _check_and_create_index(self):
        # 기존 컬렉션의 인덱스 확인 및 생성
        collection = get_collection(self.collection_name)
        if not collection.has_index():
            self._create_index(collection)
            logger.info(f"기존 컬렉션에 인덱스 생성 완료: {self.collection_name}")
        else:
            logger.info(f"컬렉션 및 인덱스가 이미 존재함: {self.collection_name}")

    def _create_index(self, collection):
        # 인덱스 생성
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index("embedding", index_params)

    def add_texts(self, texts: List[str], urls: List[str] = None):
        # 텍스트를 벡터 저장소에 추가
        collection = get_collection(self.collection_name)
        embeddings = [self.embedding_function(text) for text in texts]
        
        if urls is None or len(urls) == 0:
            urls = [""] * len(texts)
        elif len(urls) < len(texts):
            urls = urls + [""] * (len(texts) - len(urls))
        
        created_at = int(datetime.now().timestamp())
        entities = [texts, urls, embeddings, [created_at] * len(texts)]
        collection.insert(entities)
        collection.flush()
        logger.info(f"{len(texts)}개의 텍스트를 컬렉션에 추가함")
        
    def add_company_info(self, company_name: str, info: CompanyInfo):
        # 회사 정보를 벡터 저장소에 추가
        text = f"Company: {company_name}\n{info.json()}"
        url = f"company:{business_name}"
        self.add_texts([text], [f"company:{company_name}"])
    
    def add_support_program_info(self, program: SupportProgramInfo):
        # 지원 프로그램 정보를 벡터 저장소에 추가
        text = f"Support Program: {program.name}\n{program.json()}"
        self.add_texts([text], [f"program:{program.name}"])

    def search_with_similarity_threshold(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        # 유사도 임계값을 적용한 검색 수행
        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[self.embedding_function(query)],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["content", "url", "created_at"],
        )

        if not results or len(results[0]) == 0:
            logger.info("벡터 저장소에서 결과를 찾지 못함")
            return []

        hits = []
        for hit in results[0]:
            distance = hit.distance
            similarity = 1 - (distance / max(results[0][0].distance, 1))
            if similarity >= threshold:
                hits.append({
                    "content": hit.entity.get("content"),
                    "url": hit.entity.get("url"),
                    "created_at": hit.entity.get("created_at"),
                    "metadata": {"similarity": similarity}
                })

        if not hits:
            logger.info(f"유사도 임계값 {threshold}를 충족하는 결과가 없음")

        return hits

    def search_by_date_range(self, query: str, start_date: datetime, end_date: datetime, k: int = 5) -> List[Dict[str, Any]]:
        # 날짜 범위를 지정하여 검색 수행
        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[self.embedding_function(query)],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["content", "url", "created_at"],
            expr=f"created_at >= {int(start_date.timestamp())} && created_at <= {int(end_date.timestamp())}"
        )

        if not results or len(results[0]) == 0:
            logger.info("지정된 날짜 범위에서 결과를 찾지 못함")
            return []

        hits = []
        for hit in results[0]:
            hits.append({
                "content": hit.entity.get("content"),
                "url": hit.entity.get("url"),
                "created_at": datetime.fromtimestamp(hit.entity.get("created_at")),
                "metadata": {"distance": hit.distance}
            })

        return hits

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
    
    def add_data_to_partitions(self, texts: Dict, partition_name: str):
        """
        특정 파티션에 데이터를 추가
        : param texts : 추가할 텍스트 리스트
        : param partition_name : 데이터를 추가할 파티션 이름
        """
        logger.info(f"add_data_to_partitions 진입 시 partition_name: {partition_name}")
        collection = get_collection(self.collection_name)


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


    def search_in_partition(self, query: str, partition_name: str, k: int = 50, business_id: int = None)-> List[Dict[str, str]]:
        """
        특정 파티션에서 유사도 검색 수행
        :param query : 검색 쿼리
        :param partition_name: 검색할 파티션 이름
        :param k : 반환할 결과 수
        :param businessId : 조회할 businessId
        :return : 유사한 문서 리스트
        """

        collection = get_collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # if(business_id):
        #     expr = f'content like "%\\"businessId\\": {business_id}%"'

        #     filtered_results = collection.query(
        #         expr=expr, 
        #         partition_names=[partition_name], 
        #         output_fields=["content", "embedding"]
        #     )
       
        #     if not filtered_results:
        #         logger.info(f"{partition_name} 파티션에서 businessId {business_id}에 해당하는 데이터를 찾을 수 없습니다.")
        #         return []
            
        #     # 필터링된 데이터를 바탕으로 유사도 검색 수행
        #     search_results = []

        #     for item in filtered_results:
        #         embedding = self.embedding_function(item["content"])
        #         result = collection.search(
        #             data=[embedding],
        #             anns_field="embedding",
        #             param=search_params,
        #             limit=k,
        #             partition_names=[partition_name],
        #             output_fields=["content"]
        #         )
        #         search_results.extend(result)

        # else :
             # 기본 검색
        search_results = collection.search(
                data=[self.embedding_function(query)],
                anns_field="embedding",
                param=search_params,
                limit=k,
                partition_names=[partition_name],
                output_fields=["content"]
        )

        
        if not search_results or len(search_results [0]) == 0 :
                logger.info(f"{partition_name}의 이름의 파티션의 데이터를 찾을 수 없습니다.")
                return []
        
        # 서버 측에서 businessId로 필터링
        filtered_hits = []
        for hit in search_results[0]:
            content = hit.entity.get("content")
            if content and (business_id is None or f'"businessId": {business_id}' in content):
                filtered_hits.append({
                    "content": content,
                    "similarity": 1 - (hit.distance / max(search_results[0][0].distance, 1))
                })


        # hits = [
        #     {
        #         "content": hit.entity.get("content"),
        #         "similarity": 1 - (hit.distance / max(search_results[0][0].distance, 1))
        #     }
        #     for hit in search_results[0]
        #     if hit.entity.get("content")
        # ]

        # if not hits : 
        #     logger.info(f"{partition_name} 파티션에서 검색 결과를 찾을 수 없습니다.")
        
        # return hits
        if not filtered_hits:
            logger.info(f"{partition_name} 파티션에서 검색 결과를 찾을 수 없습니다.")
    
        return filtered_hits
    
    def search_with_similarity_without_url(self, query: str, k: int = 20, threshold: float = 0.4) -> List[Dict[str, str]]:
        """
        유사도 임계값을 적용한 검색 수행 (url 필드 없이)
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
            output_fields=["content"],  # url 필드를 제외
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
                    "metadata": {"similarity": similarity}
                })

        if not hits:
            logger.info(f"No results met the similarity threshold of {threshold}.")

        return hits
    
    def get_data_by_business_id(self, business_id: int) -> Dict[str, List[Dict[str, str]]]:
        """
        특정 business_id에 해당하는 데이터를 모든 파티션에서 검색
        :param business_id: 조회할 business_id
        :return: 각 파티션에서 찾은 데이터의 딕셔너리
        """
        partition_names = ["BusinessInfo", "Question", "TransactionFile", "IncomeTaxFile"]
        results = {}

        for partition_name in partition_names:
            result = self.search_in_partition(query="", partition_name=partition_name, business_id=business_id)
            results[partition_name] = result

        return results
    
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