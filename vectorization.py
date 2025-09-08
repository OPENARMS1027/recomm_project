import json
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sentence_transformers import SentenceTransformer   

# 파일 경로 설정
data_dir = './clothes'

# SBERT 모델 로드
print("1. SBERT 모델 불러오기:")
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# JSON 파일에서 상품 데이터 불러오기
def load_all_items(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                all_data.extend(json_data)
    return all_data

# 상품 텍스트(상품명)를 SBERT 벡터로 변환
def vectorize_text(item_data, model):
    text = item_data.get('name', '')
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())   
    
    return model.encode(text)   # 문장 단위 임베딩

# 카테고리 특징을 one-hot 인코딩 벡터로 변환
def vectorize_categorical(item_data, vectorizer):
    features = [{
        'brand': item_data.get('brand', ''),
        'gender': item_data.get('gender', ''),
        'category': item_data.get('category', '')
    }]
    return vectorizer.transform(features).toarray().flatten()

# 최종 결합 벡터 생성
def get_final_vectors(items, sbert_model):
    final_vectors = {}

    # 모든 아이템의 카테고리, 성별, 브랜드 정보를 DictVectorizer에 학습
    categorical_features = [{
        'brand': item.get('brand', ''),
        'gender': item.get('gender', ''),
        'category': item.get('category', '')
    } for item in items]

    categorical_vectorizer = DictVectorizer(sparse=False)
    categorical_vectorizer.fit(categorical_features)

    for item in items:
        item_id = item.get('id')

        # 텍스트 벡터 (SBERT)
        text_vector = vectorize_text(item, sbert_model)

        # 카테고리 벡터 (One-hot)
        categorical_vector = categorical_vectorizer.transform([{
            'brand': item.get('brand', ''),
            'gender': item.get('gender', ''),
            'category': item.get('category', '')
        }]).flatten()

        # 최종 결합
        combined_vector = np.concatenate((text_vector, categorical_vector))
        final_vectors[item_id] = combined_vector

    return final_vectors

# 실행 흐름
print('2. 상품 데이터 로드')
all_items = load_all_items(data_dir)

print("3. 최종 결합 벡터 생성")
item_vectors = get_final_vectors(all_items, sbert_model)

print("\n--- 최종 벡터화 결과 예시 ---")
if item_vectors:
    first_item_id = next(iter(item_vectors))
    sample_vector = item_vectors[first_item_id]
    print(f"상품 ID: {first_item_id}")
    print(f"최종 벡터 차원: {sample_vector.shape}")
    print(f"최종 벡터 (일부):\n{sample_vector[:10]}...")
