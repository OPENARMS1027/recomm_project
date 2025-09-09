import numpy as np
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "clothes_recommendation"
COLLECTION_NAME = "items"

def get_user_profile_vector(liked_item_ids: list):
    client = MongoClient(MONGO_URI)
    try:
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # 찜한 리스트에 해당하는 상품(DB에 넣은) 가져오기
        db_items = collection.find({"_id": {"$in": liked_item_ids}}, {"vector": 1})
        
        # DB에 저장된 벡터 리스트를 numpy 배열로 변환
        item_vectors_from_db = [np.array(item.get("vector")) for item in db_items]
        
        if item_vectors_from_db:
            # 찜한 상품 벡터들의 평균을 계산하여 사용자 프로필 벡터 생성
            user_profile_vector = np.mean(item_vectors_from_db, axis=0)
            return user_profile_vector
        else:
            # 찜한 상품이 없으면 0으로 채워진 벡터 반환
            # SBERT 모델 차원과 일치 필요
            return np.zeros('여기')
            
    finally:
        client.close()