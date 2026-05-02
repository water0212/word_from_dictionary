import json
import itertools
import random
from collections import defaultdict
import time
import os

# === 檔案設定 ===
# 把這裡換成你電腦裡真實的路徑
FILES = {
    'train': r'C:\Users\小菜\Desktop\台灣南島語新詞專題\word_from_dictionary\multidivide\dataset_train.txt',
    'valid': r'C:\Users\小菜\Desktop\台灣南島語新詞專題\word_from_dictionary\multidivide\dataset_valid.txt',
    'test':  r'C:\Users\小菜\Desktop\台灣南島語新詞專題\word_from_dictionary\multidivide\dataset_test.txt'
}
PRIORITY = {'train': 1, 'valid': 2, 'test': 3}

# === 核心邏輯參數 ===
BASE_UNIT = 13000

# 四個任務定義：(正樣本Level集合, 負樣本Level集合)
TASKS = {
    "task_01_vs_2345": ([0, 1], [2, 3, 4, 5]),
    "task_012_vs_345": ([0, 1, 2], [3, 4, 5]),
    "task_0123_vs_45": ([0, 1, 2, 3], [4, 5]),
    "task_01234_vs_5": ([0, 1, 2, 3, 4], [5])
}

word_split = {}
word_codes = defaultdict(set)

def get_levels(code):
    code = code.ljust(7, 'X')
    return (code[0], code[1], code[2:4], code[4], code[5:7])

def calc_distance(c1, c2):
    l1, l2 = get_levels(c1), get_levels(c2)
    if l1[0] != l2[0]: return 5
    if l1[1] != l2[1]: return 4
    if l1[2] != l2[2]: return 3
    if l1[3] != l2[3]: return 2
    if l1[4] != l2[4]: return 1
    return 0

def min_distance(w1, w2):
    min_dist = 5 
    codes_w1, codes_w2 = list(word_codes[w1]), list(word_codes[w2])
    best_c1, best_c2 = codes_w1[0], codes_w2[0]
    for c1 in codes_w1:
        for c2 in codes_w2:
            dist = calc_distance(c1, c2)
            if dist < min_dist: 
                min_dist, best_c1, best_c2 = dist, c1, c2
            if min_dist == 0: return 0, best_c1, best_c2 
    return min_dist, best_c1, best_c2

def determine_target(w1, w2):
    s1, s2 = word_split[w1], word_split[w2]
    if s1 == 'test' or s2 == 'test': return 'test'
    if s1 == 'valid' or s2 == 'valid': return 'valid'
    return 'train'

# 1. 讀取資料
print("📖 讀取資料集中...")
for split_name, filepath in FILES.items():
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            raw_code = parts[0]
            clean_code = raw_code[:-1] if not raw_code[-1].isalnum() else raw_code
            for w in parts[1:]:
                word_codes[w].add(clean_code)
                if w not in word_split or PRIORITY[split_name] > PRIORITY[word_split[w]]:
                    word_split[w] = split_name

all_words = list(word_split.keys())

# 2. 預先蒐集 Level 0~3 的所有可能 Candidate (優化效能)
print("🔍 預處理 Level 0~3 候選池...")
candidate_pools = {split: {lv: [] for lv in range(6)} for split in ['train', 'valid', 'test']}
prefixes = {0: 7, 1: 5, 2: 4, 3: 2}

for lv in range(4):
    p_len = prefixes[lv]
    groups = defaultdict(set)
    for w, codes in word_codes.items():
        for c in codes: groups[c[:p_len]].add(w)
    
    for pfx, words in groups.items():
        if len(words) < 2: continue
        for w1, w2 in itertools.combinations(words, 2):
            dist, c1, c2 = min_distance(w1, w2)
            if dist == lv:
                target = determine_target(w1, w2)
                candidate_pools[target][lv].append((w1, c1, w2, c2))

# 3. 遍歷任務並依照動態 K 邏輯產生資料
for task_name, (pos_lvs, neg_lvs) in TASKS.items():
    print(f"\n======================================")
    print(f"🚀 啟動任務: {task_name}")
    os.makedirs(task_name, exist_ok=True)
    
    K = len(pos_lvs)
    M = len(neg_lvs)
    
    test_total = BASE_UNIT * K
    train_total = test_total * 8
    
    pos_quota = {'train': train_total // K, 'valid': test_total // K, 'test': test_total // K}
    neg_quota = {'train': train_total // M, 'valid': test_total // M, 'test': test_total // M}
    
    print(f"📊 配額計畫 (1:1 平衡, K={K}):")
    print(f"   Train(8) -> 正樣本: {train_total:,} | 負樣本: {train_total:,}")
    print(f"   Test(1)  -> 正樣本: {test_total:,} | 負樣本: {test_total:,}")
    
    # 建立一個字典來儲存這個任務的最終統計數據
    task_summary = {}

    for split in ['train', 'valid', 'test']:
        final_records = []
        
        # 處理正樣本 (Label 1)
        for lv in pos_lvs:
            target_q = pos_quota[split]
            if lv <= 3:
                available = len(candidate_pools[split][lv])
                sample = random.sample(candidate_pools[split][lv], min(available, target_q))
            else:
                sample_set = set()
                while len(sample_set) < target_q:
                    w1, w2 = random.sample(all_words, 2)
                    if determine_target(w1, w2) == split:
                        dist, c1, c2 = min_distance(w1, w2)
                        if dist == lv: sample_set.add((w1, c1, w2, c2))
                sample = list(sample_set)
            
            for r in sample:
                final_records.append({"word1": r[0], "code1": r[1], "word2": r[2], "code2": r[3], "distance": lv, "label": 1})

        # 處理負樣本 (Label 0)
        for lv in neg_lvs:
            target_q = neg_quota[split]
            if lv <= 3:
                available = len(candidate_pools[split][lv])
                sample = random.sample(candidate_pools[split][lv], min(available, target_q))
            else:
                sample_set = set()
                while len(sample_set) < target_q:
                    w1, w2 = random.sample(all_words, 2)
                    if determine_target(w1, w2) == split:
                        dist, c1, c2 = min_distance(w1, w2)
                        if dist == lv: sample_set.add((w1, c1, w2, c2))
                sample = list(sample_set)
            
            for r in sample:
                final_records.append({"word1": r[0], "code1": r[1], "word2": r[2], "code2": r[3], "distance": lv, "label": 0})

        # 計算目前的實際寫入數量
        pos_count = sum(1 for r in final_records if r["label"] == 1)
        neg_count = sum(1 for r in final_records if r["label"] == 0)
        task_summary[split] = {"total": len(final_records), "pos": pos_count, "neg": neg_count}

        # 打亂後寫入
        random.shuffle(final_records)
        save_path = os.path.join(task_name, f"{split}.jsonl")
        with open(save_path, 'w', encoding='utf-8') as f:
            for rec in final_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
    # 👉 任務結束時，印出精確的數量驗證報告
    print(f"\n   📈 【{task_name}】 數量驗證報告：")
    for s in ['train', 'valid', 'test']:
        print(f"      - {s.upper():<5} -> 總計: {task_summary[s]['total']:>7,} 筆 | 近義(1): {task_summary[s]['pos']:>7,} | 非近義(0): {task_summary[s]['neg']:>7,}")

print("\n🎉 四大任務資料集已全數產生完畢！")