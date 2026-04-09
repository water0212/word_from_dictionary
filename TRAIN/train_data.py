import json
import numpy as np
import os
from gensim.models import fasttext
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
current_dir = os.path.dirname(os.path.abspath(__file__))
fasttext_model_path = os.path.join(current_dir, "cc.zh.300.bin")
train_file = os.path.join(current_dir, "model_datasets", "train_data.jsonl")
test_file = os.path.join(current_dir, "model_datasets", "test_data.jsonl")
try:
    # 使用 Gensim 載入 FastText 模型
    ft_model = fasttext.load_facebook_model(fasttext_model_path)
    print("載入成功")
except FileNotFoundError:
    print(f"找不到模型檔案 {fasttext_model_path}")
    exit()

def extract_features(word1, word2, model):
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    cos_sim = model.wv.similarity(word1, word2)
    
    
    abs_diff = np.abs(vec1 - vec2)
    
    #vec1(300) + vec2(300) + abs_diff(300) + cos_sim(1)
    features = np.concatenate([vec1, vec2, abs_diff, np.array([cos_sim, euclidean_dist])])
    
    return features

def load_data(file_path, model):
    data = []
    X, Y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            word1 = data['word1']
            word2 = data['word2']
            score = data['score']
            features = extract_features(word1, word2, model)
            X.append(features)
            Y.append(score)
    return np.array(X), np.array(Y)

print("\n📊 正在處理訓練集資料...")
X_train, Y_train = load_data(train_file, ft_model)
print(f"訓練集大小: {X_train.shape}")

print("📊 正在處理測試集資料...")
X_test, Y_test = load_data(test_file, ft_model)
print(f"測試集大小: {X_test.shape}")