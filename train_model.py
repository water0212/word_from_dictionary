import json
import numpy as np
import time
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial.distance import cosine

# 1. 快速載入 FastText 模型
KV_LOAD_PATH = 'fasttext_zh.kv'
print("⚡ 開始快速載入模型...")
start_time = time.time()
wv = KeyedVectors.load(KV_LOAD_PATH, mmap='r')
print(f"🚀 載入成功！耗時: {time.time() - start_time:.4f} 秒")

# 2. 定義 901 維度的特徵萃取函數
def extract_features(word1, word2, wv):
    try:
        v1 = wv[word1]
    except KeyError:
        v1 = np.zeros(300)
        
    try:
        v2 = wv[word2]
    except KeyError:
        v2 = np.zeros(300)
        
    v_diff = v1 - v2
    
    if np.all(v1 == 0) or np.all(v2 == 0):
        cos_sim = 0.0
    else:
        cos_sim = 1.0 - cosine(v1, v2)
        
    features = np.concatenate([v1, v2, v_diff, [cos_sim]])
    return features

# 3. 定義 JSON 資料讀取與轉換函數
def load_and_process_data(filepath, wv):
    X = []
    y = []
    print(f"處理檔案: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for label, pairs in data.items():
            for pair in pairs:
                w1 = pair['word1']
                w2 = pair['word2']
                features = extract_features(w1, w2, wv)
                
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

# 4. 準備資料集
X_train, y_train = load_and_process_data('train.json', wv)
X_valid, y_valid = load_and_process_data('valid.json', wv)
# X_test, y_test = load_and_process_data('test.json', wv) # 若有測試集可取消註解

# 5. 建立與訓練模型
print("開始訓練分類模型 (Random Forest)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train)

# 6. 進行驗證與評估
print("進行驗證集預測...")
y_pred = clf.predict(X_valid)

# ================= 新增：儲存至 Excel 區塊 =================

# 取得字典格式的評估報告
report_dict = classification_report(y_valid, y_pred, output_dict=True)

# 將字典轉換為 Pandas DataFrame
# .transpose() 是為了讓排版更符合一般閱讀習慣 (類別在列，指標在欄)
df_report = pd.DataFrame(report_dict).transpose()

# 取得整體的 Accuracy 並印出
acc = accuracy_score(y_valid, y_pred)
print(f"整體準確率 (Accuracy): {acc:.4f}")

# 設定輸出的 Excel 檔名
excel_filename = "evaluation_results.xlsx"

# 將資料寫入 Excel
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 第一個 Sheet：詳細的分類報告
    df_report.to_excel(writer, sheet_name='Classification_Report')
    
    # 你也可以把其他總結資訊寫進另一個 Sheet (選用)
    # df_summary = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [acc]})
    # df_summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"✅ 評估數據已成功儲存至 {excel_filename}")