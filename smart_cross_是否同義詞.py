import json
import itertools
import random
from collections import defaultdict
import time
import os

# === 檔案路徑設定 ===
FILES = {
    'train': './divide_dataset_output/dataset_train.txt',
    'valid': './divide_dataset_output/dataset_valid.txt',
    'test': './divide_dataset_output/dataset_test.txt'
}

# 輸出路徑改為 .jsonl
OUTPUT_PATHS = {
    'train': './TRAIN/dataset_yes_or_no/train.jsonl',
    'valid': './TRAIN/dataset_yes_or_no/valid.jsonl',
    'test': './TRAIN/dataset_yes_or_no/test.jsonl'
}

PRIORITY = {'train': 1, 'valid': 2, 'test': 3}

# 數量分配邏輯
QUOTA_SYNONYM = 130000               # 正樣本 (Level 0) 總數
QUOTA_NON_SYNONYM_PER_LEVEL = 26000  # 負樣本 (Level 1~5) 每個階級抽 2.6 萬組，共 13 萬組

word_split = {}
word_codes = defaultdict(set)

def get_levels(code):
    code = code.ljust(7, 'X')
    return (code[0], code[1], code[2:4], code[4], code[5:7])

def calc_distance(c1, c2):
    """計算距離：0=完全相同, 5=完全相異"""
    l1, l2 = get_levels(c1), get_levels(c2)
    if l1[0] != l2[0]: return 5
    if l1[1] != l2[1]: return 4
    if l1[2] != l2[2]: return 3
    if l1[3] != l2[3]: return 2
    if l1[4] != l2[4]: return 1
    return 0

def min_distance(w1, w2):
    """尋找兩個詞之間的最短距離，並回傳距離與對應代碼"""
    min_dist = 5 
    codes_w1 = list(word_codes[w1])
    codes_w2 = list(word_codes[w2])
    best_c1 = codes_w1[0]
    best_c2 = codes_w2[0]

    for c1 in codes_w1:
        for c2 in codes_w2:
            dist = calc_distance(c1, c2)
            if dist < min_dist: 
                min_dist = dist
                best_c1 = c1
                best_c2 = c2
            if min_dist == 0: 
                return 0, best_c1, best_c2 
                
    return min_dist, best_c1, best_c2

# 步驟 1: 讀取資料集
print("📖 開始讀取資料集...")
for split_name, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到檔案: {filepath}，跳過。")
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            raw_code = parts[0]
            clean_code = raw_code[:-1] if not raw_code[-1].isalnum() else raw_code
            for w in parts[1:]:
                word_codes[w].add(clean_code)
                current_split = word_split.get(w, 'train')
                if w not in word_split or PRIORITY[split_name] > PRIORITY[current_split]:
                    word_split[w] = split_name

all_words = list(word_split.keys())
print(f"✅ 讀取完成！共有 {len(all_words)} 個不重複詞彙。\n")

collected_pairs_by_dist = {i: [] for i in range(6)}
start_time = time.time()

# 步驟 2: 精準撈取 Level 0 ~ 3 (稀有組合)
print("🔍 開始精準撈取 Level 0 ~ 3 (稀有組合)...")
prefixes = {0: 7, 1: 5, 2: 4, 3: 2}

for target_dist in range(0, 4):
    prefix_len = prefixes[target_dist]
    current_quota = QUOTA_SYNONYM if target_dist == 0 else QUOTA_NON_SYNONYM_PER_LEVEL
    
    groups = defaultdict(set)
    for w, codes in word_codes.items():
        for c in codes:
            groups[c[:prefix_len]].add(w)
            
    temp_pairs = set()
    for pfx, words_in_group in groups.items():
        if len(words_in_group) < 2: continue
        for w1, w2 in itertools.combinations(words_in_group, 2):
            dist, _, _ = min_distance(w1, w2)
            if dist == target_dist:
                temp_pairs.add(tuple(sorted((w1, w2))))
                
    temp_list = list(temp_pairs)
    if len(temp_list) > current_quota:
        selected = random.sample(temp_list, current_quota)
    else:
        selected = temp_list
    
    collected_pairs_by_dist[target_dist] = selected
    print(f"   ➤ Level {target_dist} 收集完畢：{len(selected):,} 組")

# 步驟 3: 高速盲抽 Level 4 ~ 5 (氾濫組合)
print("\n🎲 開始高速盲抽 Level 4 ~ 5 (氾濫組合)...")
for target_dist in [4, 5]:
    current_quota = QUOTA_NON_SYNONYM_PER_LEVEL
    selected_set = set()
    while len(selected_set) < current_quota:
        w1, w2 = random.sample(all_words, 2)
        dist, _, _ = min_distance(w1, w2)
        if dist == target_dist:
            selected_set.add(tuple(sorted((w1, w2))))
    
    collected_pairs_by_dist[target_dist] = list(selected_set)
    print(f"   ➤ Level {target_dist} 收集完畢：{current_quota:,} 組")

# 步驟 4: 分發並寫入 JSONL
print("\n📝 準備匯出 JSONL 檔案...")

# 開啟檔案準備寫入
out_files = {k: open(v, 'w', encoding='utf-8') for k, v in OUTPUT_PATHS.items()}

total_count = 0
# 為了讓訓練更穩定，建議將所有蒐集到的配對混合後再寫入 (避免檔案開頭全是 Level 0)
# 我們先暫存在記憶體中進行分發
all_records = {'train': [], 'valid': [], 'test': []}

for dist, pairs in collected_pairs_by_dist.items():
    label = 1 if dist == 0 else 0
    for w1, w2 in pairs:
        _, c1, c2 = min_distance(w1, w2)
        s1, s2 = word_split[w1], word_split[w2]
        
        # 優先級分發
        if s1 == 'test' or s2 == 'test':
            target = 'test'
        elif s1 == 'valid' or s2 == 'valid':
            target = 'valid'
        else:
            target = 'train'
            
        record = {
            "word1": w1, "code1": c1,
            "word2": w2, "code2": c2,
            "distance": dist, "label": label
        }
        all_records[target].append(record)

# 隨機打亂每個集，然後寫入 JSONL
for split in ['train', 'valid', 'test']:
    random.shuffle(all_records[split])
    for item in all_records[split]:
        out_files[split].write(json.dumps(item, ensure_ascii=False) + '\n')
        total_count += 1
    out_files[split].close()

elapsed = time.time() - start_time
print(f"\n🎉 大功告成！總共匯出 {total_count:,} 筆 JSONL 資料。")
print(f"耗時: {elapsed:.2f} 秒")