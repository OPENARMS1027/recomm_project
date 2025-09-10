# sessionstorage데이터 기반 사용자 벡터화

import numpy as np
import os
from pymongo import MongoClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MONGO_URI = os.getenv("DB_MONGO_URI")
DB_NAME = "crawl"
COLLECTION_NAME = "items"

# 벡터 차원 상수 정의 필요
VECTOR_DIMENSION = ''

# DB에서 DB에서 상품 벡터 들고오기
def get_item_vector_from_db(item_id: str):
    client = MongoClient(MONGO_URI)
    try:
        collection = client[DB_NAME][COLLECTION_NAME]
        item = collection.find_one({"_id": item_id})
        if item and 'vector' in item:
            return np.array(item['vector'])
        return None
    finally:
        client.close()



# 프론트에 세션에있는 데이터 넣어줘서 사용자 벡터화 하는 함수임
app = FastAPI()
class UserAction(BaseModel):
    userId: str
    timestamp: str
    actionType: str
    itemId: Optional[str] = None
    searchTerm: Optional[str] = None

@app.post("/users/profile-vector-from-session")
def create_user_profile_vector(actions:List[UserAction]):
    if not actions:
        return {"user_profile_vector": np.zeros(VECTOR_DIMENSION).tolist(),
                "message": "전달된 사용자 행동 데이터가 없습니다."
                }
    
    weighted_vectors = []

    for action in actions:
        vector = None
        weight = 0
        
        # 행동 타입에 따라 벡터를 가져오고 가중치를 부여
        if action.actionType == 'like':
            vector = get_item_vector_from_db(action.itemId)
            # weight = 5 # 찜에 대한 높은 가중치
        # elif action.actionType == 'goDetail':
        #     vector = get_item_vector_from_db(action.itemId)
        #     weight = 2 # 상세 페이지 이동에 대한 중간 가중치
        # elif action.actionType == 'search':
        #     vector = vectorize_search_term(action.searchTerm)
        #     weight = 1 # 검색에 대한 낮은 가중치
        
        # 이 부분 추후에 계산 수정이 필요하지 않을까 함!!
        if vector is not None and weight > 0:
            weighted_vectors.append(vector)
            # weighted_vectors.append(vector * weight)

    if weighted_vectors:
        user_profile_vector = np.mean(weighted_vectors, axis=0)
        try:
            client = MongoClient(MONGO_URI)
            collection = client[DB_NAME][COLLECTION_NAME]
            user_id = actions[0].userId # 첫 번째 행동에서 사용자 ID를 가져옴
            db_doc = {
                "_id": user_id,
                "profile_vector": user_profile_vector.tolist(),
            }
            collection.update_one({"_id": user_id}, {"$set": db_doc}, upsert=True)
            return {"user_profile_vector": user_profile_vector.tolist(), "message": "사용자 벡터가 성공적으로 저장되었습니다."}
        except Exception as e:
            return {"error": "DB 저장 중 오류 발생", "details": str(e)}
        finally:
            client.close()
    else:
        return {"message": "찜 목록이 없는 상태로 벡터가 저장되었씁니다!"}

create_user_profile_vector()