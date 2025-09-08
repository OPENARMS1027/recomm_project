import numpy as np
from pymongo import MongoClient
from pymongo import UpdateOne

# 벡터화된 데이터
vector_data = ''

# 교체 필요
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "clothes_recommendation"
COLLECTION_NAME = "items"


#  mongoDB에 저장하는 함수
def save_vectors_to_mongo(item_vectors: dict):
    """
    벡터
    item_vectors (dict): 상품 ID를 키로, 최종 벡터를 값으로 하는 딕셔너리
    이거는 벡터화한 데이터임
    """
    try:
        # MongoDB 클라이언트 연결
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # 상품 벡터를 문서(Document) 형태로 변환하여 일괄 처리
        operations = []
        for item_id, vector_data in item_vectors.items():
            # NumPy 배열을 MongoDB에 저장 가능한 파이썬 리스트로 변환
            vector_list = vector_data.tolist()
            
            # MongoDB 문서 형식에 맞춰 데이터 준비
            document = {
                "_id": item_id,
                "vector": vector_list,
            }
            # _id가 존재하면 업데이트, 없으면 삽입 (upsert=True)
            operations.append(
                UpdateOne({"_id": document["_id"]}, {"$set": document}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations, ordered=False)
            
        print(f"✅ {len(item_vectors)}개의 상품 벡터가 MongoDB에 성공적으로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ MongoDB 저장 중 오류 발생: {e}")
    finally:
        # 연결 종료
        client.close()

# ⚠️ 사용 예시 (item_vectors 딕셔너리가 이미 생성되었다고 가정)
# 예시 더미 데이터:
# item_vectors = {
#     "123": np.random.rand(384),
#     "456": np.random.rand(384),
# }
# save_vectors_to_mongo(item_vectors)