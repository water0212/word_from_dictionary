import json
import os

# ==========================================
# 1. 路徑與參數設定
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, "experiment_pairs.json")
output_dir = os.path.join(current_dir, "model_datasets")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 定義分數對應表 (陣列格式)
# ==========================================
# 這裡採用「距離分數」的概念：數字越小代表越相似 (0=同義, 5=完全無關)
# 您可以隨時依據教授的需求，將其改為「相似度分數」(例如 5.0 到 0.0)
score_mapping = {
    "Synonym": 1,
    "D1": 0.92,
    "D2": 0.8,
    "D3": 0.58,
    "D4": 0.29,
    "D5": 0.0
}

def main():
    print("讀取原始配對資料中...")
    if not os.path.exists(input_file):
        print(f"找不到檔案: {input_file}，請確認路徑是否正確。")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 準備三個資料集的容器
    datasets = {
        "Train": [],
        "Valid": [],
        "Test": []
    }

    # ==========================================
    # 3. 解析巢狀 JSON 並扁平化
    # ==========================================
    print("正在轉換格式與標記分數...")
    for letter, splits in raw_data.items():
        for split_name, categories in splits.items():
            # 確保只處理我們定義好的三大資料集
            if split_name not in datasets:
                continue
                
            for cat_name, pairs_dict in categories.items():
                # 取得對應的分數陣列，若發生意外類別則預設給 [99.0] 作為防呆
                score_array = score_mapping.get(cat_name, [99.0])
                
                for key, words in pairs_dict.items():
                    if len(words) == 2:
                        # 建立單筆訓練資料的結構
                        record = {
                            "word1": words[0],
                            "word2": words[1],
                            "score": score_array,
                            "category": cat_name  # 保留原始類別方便後續 debug
                        }
                        datasets[split_name].append(record)

    # ==========================================
    # 4. 輸出成 JSONL 格式 (機器學習最常用的格式)
    # ==========================================
    print("\n" + "="*50)
    print("🚀 開始輸出資料集...")
    print("="*50)
    
    for split_name, records in datasets.items():
        # 輸出檔名例如: train_data.jsonl
        out_path = os.path.join(output_dir, f"{split_name.lower()}_data.jsonl")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for record in records:
                # json.dumps 確保每一行都是一個獨立的 JSON 物件
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"✅ {split_name} 資料集已儲存: 包含 {len(records)} 筆配對 -> {out_path}")

if __name__ == "__main__":
    main()
