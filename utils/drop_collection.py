from pymilvus import utility, Collection
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector_store import VectorStore
from config.settings import settings
from typing import List, Dict



from utils.database import connect_to_milvus, get_collection

def drop_all_collection():
    """
    Milvus 모든 컬렉션 삭제
    """
    collections = utility.list_collections()
    for collection_name in collections :
        # utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' 삭제됨")
    print("모든 컬렉션 삭제완료")

def drop_one_collection(collection_name):
    """
    특정 Milvus 컬렉션 삭제
    :param collection_name : 삭제할 컬렉션의 이름
    """

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' 삭제됨")
    else:
        print(f"Collection '{collection_name}'이(가) 존재하지 않습니다.")

def get_all_collection() :
    collections = utility.list_collections()
    for collection_name in collections : 
        print(f"Collection : {collection_name}")

def drop_collection_indexes(collection_name) :
    vector_store = VectorStore()
    vector_store.use_custom_collection(collection_name)

    

    try:
        vector_store.delete_all_indexs()
        print(f"Collection {collection_name}의 모든 인덱스가 삭제되었습니다.")
    except Exception as e:
        print(f"인덱스를 삭제하는 과정 중 오류가 발생했습니다 : {str(e)}")

def drop_partition_indexes(collection_name : str, partition_name : str) : 
    """
    파티션의 모든 데이터 삭제
    """
    vector_store = VectorStore()
    vector_store.use_custom_collection(collection_name)

    try:
        collection = Collection(collection_name)
        expr = "id >= 0" # 조건 
        collection.delete(expr, partition_name=partition_name)
        collection.flush()
        print(f"{partition_name} 파티션의 모든 데이터를 삭제하였습니다.")

    except Exception as e :
        print(f"{partition_name} 파티션의 모든 데이터를 삭제하는 중 오류가 발생했습니다. : {str(e)}")



def get_collection_fields(collection_name):
    collection = Collection(collection_name)
    fields = collection.schema.fields
    
    for field in fields : 
        print(f"Field name: {field.name}, Type: {field.dtype}")

def get_sample_data(collection_name: str, limit: int = 5):
    """
    컬렉션에서 샘플 데이터를 조회하는 함수
    :param collection_name: 컬렉션 이름
    :param limit: 조회할 데이터 개수
    :return: 조회된 데이터
    """

    collection = get_collection(collection_name)  # 컬렉션 가져오기
    
    # 컬렉션에 있는 모든 파티션 가져오기
    partitions = collection.partitions
    print(f"Collection '{collection_name}'의 파티션 목록: ")
    for partition in partitions:
        print(f"- {partition.name}")

    results = []
    for partition in partitions:
        print(f"Querying partition: {partition.name}")
        partition_results = collection.query(
            expr="id >= 0",
            output_fields=["id", "content", "embedding"],
            partition_names=[partition.name],
            limit=limit
        )
        print(f"Results from partition: {partition.name}")
        for result in partition_results:
            print(f"Partition: {partition.name}, Content: {result['content'][:200]}")

    for result in results:
        content_preview = result["content"][:200] + "..." if len(result["content"]) > 10 else result["content"]
        print({
            "partition": result["partition"],
            "content": content_preview,
        })
    
    # return result

def get_data_from_partition(collection_name: str, partition_name: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    특정 파티션에서 데이터를 조회하는 메소드
    :param partition_name: 데이터를 조회할 파티션 이름
    :param limit: 조회할 데이터 개수
    :return: 조회된 데이터 리스트
    """
    collection = get_collection(collection_name)
    
    # 특정 파티션에서 데이터를 조회합니다.
    results = collection.query(
        expr="id >= 0",  # 모든 데이터를 조회
        partition_names=[partition_name],  # 조회할 파티션 이름
        output_fields=["id", "content", "embedding"],  # 원하는 필드 선택
        limit=limit
    )
    
    # 결과를 정리하여 반환합니다.
    formatted_results = [
        {
            # "id": result["id"],
            "partition": partition_name,
            "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
            # "embedding": result["embedding"][:10] + "..." if len(result["embedding"]) > 10 else result["embedding"]
        }
        for result in results
    ]

    print (formatted_results)
    
    return formatted_results

def unload_collection(collection_name):
    collection = get_collection(collection_name)

    try :
        collection.release()

        while utility.has_collection(collection_name):
            time.sleep(1)
            print(time)

        print(f"{collection_name} 컬렉션이 언로드 되었습니다.")
    except Exception as e :
        print(f"컬렉션 언로드 중 오류가 발생했습니다 : {e}")

    

connect_to_milvus()

# 컬렉션 목록 조회
# get_all_collection()
# 한 컬렉션 모든 필드 조회
# get_collection_fields(settings.COLLECTION_NAME2)

# 샘플 데이터 조회
# 한 컬렉션의 전체 데이터 조회
# get_sample_data(settings.COLLECTION_NAME2, limit=5)
# 한 파티션의 데이터 조회
get_data_from_partition(settings.COLLECTION_NAME2, "SimpleTransaction", 50 )

# 모든 컬렉션 삭제
# drop_all_collection()
# 한 컬렉션 삭제
# drop_one_collection(settings.COLLECTION_NAME2)
# 한 컬렉션의 모든 인덱스 삭제
# drop_collection_indexes(settings.COLLECTION_NAME2)
# 한 파티션의 모든 데이터 삭제
# drop_partition_indexes(settings.COLLECTION_NAME2, "SimpleTransaction")
# vectorDB unload
# unload_collection(settings.COLLECTION_NAME2)