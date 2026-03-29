import json
import itertools
import random
import os
from collections import defaultdict
import time

current_dir = os.path.dirname(os.path.abspath(__file__))

# === 請在這裡填入你的相對路徑 ===
# 輸入檔案的相對路徑
TRAIN_INPUT_PATH = os.path.join(current_dir, "divide_dataset_output", "dataset_train.txt")
VALID_INPUT_PATH = os.path.join(current_dir, "divide_dataset_output", "dataset_valid.txt")
TEST_INPUT_PATH  = os.path.join(current_dir, "divide_dataset_output", "dataset_test.txt")

# 輸出檔案的相對路徑
TRAIN_OUTPUT_PATH = os.path.join(current_dir, "smart_cross_output", "train.jsonl")
VALID_OUTPUT_PATH = os.path.join(current_dir, "smart_cross_output", "valid.jsonl")
TEST_OUTPUT_PATH  = os.path.join(current_dir, "smart_cross_output", "test.jsonl")
# =================================

# 檔案路徑設定
FILES = {
    'train': TRAIN_INPUT_PATH,
    'valid': VALID_INPUT_PATH,
    'test': TEST_INPUT_PATH
}
OUTPUT_PATHS = {
    'train': TRAIN_OUTPUT_PATH,  # 例如: 'D:/my_project/data/train.json'
    'valid': VALID_OUTPUT_PATH,  # 例如: 'D:/my_project/data/valid.json'
    'test':  TEST_OUTPUT_PATH     # 例如: 'D:/my_project/data/test.json'
}

PRIORITY = {'train': 1, 'valid': 2, 'test': 3}
QUOTA = 130000

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

# 👉 修改：讓函式同時回傳「最短距離」以及「對應的兩個代碼」
def min_distance(w1, w2):
    """尋找兩個詞之間的最短距離，並同時回傳達成該距離的代碼"""
    min_dist = 5 
    best_c1 = list(word_codes[w1])[0] # 預設先拿第一個代碼墊底
    best_c2 = list(word_codes[w2])[0]

    for c1 in word_codes[w1]:
        for c2 in word_codes[w2]:
            dist = calc_distance(c1, c2)
            if dist < min_dist: 
                min_dist = dist
                best_c1 = c1
                best_c2 = c2
            if min_dist == 0: 
                return 0, best_c1, best_c2 # 已經是 0，提早結束
                
    return min_dist, best_c1, best_c2

# 步驟 1: 讀取資料集
print("📖 開始讀取資料集...")
for split_name, filepath in FILES.items():
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                raw_code = parts[0]
                # 這裡拔掉最後面的 =, #, @，只保留純 7 碼 (例如 La06D01)
                clean_code = raw_code[:-1] if not raw_code[-1].isalnum() else raw_code
                
                for w in parts[1:]:
                    word_codes[w].add(clean_code)
                    current_split = word_split.get(w, 'train')
                    if w not in word_split or PRIORITY[split_name] > PRIORITY[current_split]:
                        word_split[w] = split_name
    except FileNotFoundError:
        print(f"⚠️ 找不到檔案: {filepath}，請確認。")
        exit()

all_words = list(word_split.keys())
print(f"✅ 讀取完成！共有 {len(all_words)} 個不重複詞彙。\n")

collected_pairs = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set()}
start_time = time.time()

# 步驟 2: 精準撈取 Level 0 ~ 3 (稀有組合)
print("🔍 開始精準撈取 Level 0 ~ 3 (稀有組合)...")
prefixes = {
    0: lambda c: c[:7], 
    1: lambda c: c[:5], 
    2: lambda c: c[:4], 
    3: lambda c: c[:2]  
}

for target_dist in range(0, 4):
    get_prefix = prefixes[target_dist]
    groups = defaultdict(set)
    for w, codes in word_codes.items():
        for c in codes:
            groups[get_prefix(c)].add(w)
            
    temp_pairs = set()
    for pfx, words_in_group in groups.items():
        if len(words_in_group) < 2: continue
        for w1, w2 in itertools.combinations(words_in_group, 2):
            # 👉 修改：這裡我們只需要取回傳值的第一個 (距離 dist) 來做判斷
            dist, _, _ = min_distance(w1, w2)
            if dist == target_dist:
                temp_pairs.add(tuple(sorted((w1, w2))))
                
    temp_list = list(temp_pairs)
    if len(temp_list) > QUOTA:
        collected_pairs[target_dist] = set(random.sample(temp_list, QUOTA))
    else:
        collected_pairs[target_dist] = set(temp_list)
        
    print(f"   ➤ Level {target_dist} 收集完畢：{len(collected_pairs[target_dist]):,} 組 (字典總上限為 {len(temp_list):,} 組)")

# 步驟 3: 高速盲抽 Level 4 ~ 5 (氾濫組合)
print("\n🎲 開始高速盲抽 Level 4 ~ 5 (氾濫組合)...")
for target_dist in [4, 5]:
    while len(collected_pairs[target_dist]) < QUOTA:
        w1, w2 = random.sample(all_words, 2)
        dist, _, _ = min_distance(w1, w2)
        if dist == target_dist:
            collected_pairs[target_dist].add(tuple(sorted((w1, w2))))
            
    print(f"   ➤ Level {target_dist} 收集完畢：{QUOTA:,} 組 (已達配額滿線)")

# 步驟 4: 依照優先級分發，並建構階層式 JSON
print("\n📝 準備匯出包含代碼的階層式 JSON 檔案...")

output_data = {
    'train': {f"level_{i}": [] for i in range(6)},
    'valid': {f"level_{i}": [] for i in range(6)},
    'test':  {f"level_{i}": [] for i in range(6)}
}

total_written = 0
for dist, pairs in collected_pairs.items():
    level_key = f"level_{dist}"
    for w1, w2 in pairs:
        
        # 👉 修改：在寫入檔案前，呼叫一次函式把兩個詞的最佳配對代碼抓出來
        _, c1, c2 = min_distance(w1, w2)
        
        s1, s2 = word_split[w1], word_split[w2]
        
        if s1 == 'test' or s2 == 'test':
            target = 'test'
        elif s1 == 'valid' or s2 == 'valid':
            target = 'valid'
        else:
            target = 'train'
            
        # 👉 修改：將代碼一起加入 JSON 的結構中
        output_data[target][level_key].append({
            "word1": w1,
            "code1": c1,
            "word2": w2,
            "code2": c2
        })
        total_written += 1

for target_name, data in output_data.items():
    save_path = OUTPUT_PATHS[target_name] 
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
print(f"\n🎉 大功告成！總共匯出 {total_written:,} 組訓練資料。耗時: {elapsed:.2f} 秒")
