import json
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sentence_transformers import SentenceTransformer

# 파일 경로 설정
data_dir = '.'

# SBERT 모델 로드
print("1. SBERT 모델 불러오기:")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# JSON 파일에서 상품 데이터 불러오기
def load_all_items(data_dir, filename):
    all_data = []
    filepath = os.path.join(data_dir, filename)
    
    # 파일이 존재하는지 확인
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_data.extend(json_data)
    else:
        print(f"오류: {filename} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        
    return all_data


# 상품 텍스트(상품명)를 SBERT 벡터로 변환
def vectorize_text(item_data, model):
    text = item_data.get('name', '')
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    
    return model.encode(text)

# 가중치 특징을 벡터로 변환하는 함수 (새롭게 정의)
def vectorize_weighted_features(item_data, all_unique_features_dict):
    vector = np.zeros(len(all_unique_features_dict))
    
    for feature_list in [item_data.get('categoryList', []), item_data.get('styleList', [])]:
        for feature in feature_list:
            name = feature.get('name')
            percentage = feature.get('percentage', 0)
            if name in all_unique_features_dict:
                index = all_unique_features_dict[name]
                vector[index] = percentage / 100.0  # 비율을 0~1 사이 값으로 변환
    return vector

# 최종 결합 벡터 생성
def get_final_vectors(items, sbert_model):
    final_vectors = {}

    # 1. 모든 아이템의 가중치 특징(category, style) 고유값 목록화
    all_unique_features = set()
    for item in items:
        for feature_list in [item.get('categoryList', []), item.get('styleList', [])]:
            for feature in feature_list:
                all_unique_features.add(feature.get('name'))
    
    all_unique_features_dict = {name: i for i, name in enumerate(sorted(list(all_unique_features)))}

    # ⚠️ 개선된 부분: DictVectorizer의 학습을 루프 밖에서 한 번만 수행
    all_brand_gender_features = [{'brand': item.get('brand', ''), 'gender': item.get('gender', '')} for item in items]
    brand_gender_vectorizer = DictVectorizer(sparse=False)
    brand_gender_vectorizer.fit(all_brand_gender_features)


    for item in items:
        item_id = item.get('id')

        # 텍스트 벡터 (SBERT)
        text_vector = vectorize_text(item, sbert_model)

        # 가중치 벡터 (새로운 함수 사용)
        weighted_vector = vectorize_weighted_features(item, all_unique_features_dict)
        
        # 기타 카테고리 정보 벡터화 (학습된 Vectorizer 사용)
        brand_gender_features = {'brand': item.get('brand', ''), 'gender': item.get('gender', '')}
        brand_gender_vector = brand_gender_vectorizer.transform([brand_gender_features]).flatten()


        # 최종 결합: 텍스트 + 가중치 특징 + 기타 카테고리 특징
        combined_vector = np.concatenate((text_vector, weighted_vector, brand_gender_vector))
        final_vectors[item_id] = combined_vector

    return final_vectors

# 실행 흐름
print('2. 상품 데이터 로드')
all_items = load_all_items(data_dir, "upper_data.json")

print("3. 최종 결합 벡터 생성")
item_vectors = get_final_vectors(all_items, sbert_model)

print("\n--- 최종 벡터화 결과 예시 ---")
if item_vectors:
    first_item_id = next(iter(item_vectors))
    sample_vector = item_vectors[first_item_id]
    print(f"상품 ID: {first_item_id}")
    print(f"최종 벡터 차원: {sample_vector.shape}")
    print(f"최종 벡터 (일부):\n{sample_vector[:10]}...")