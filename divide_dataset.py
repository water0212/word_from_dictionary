import math
import random
from collections import defaultdict

file_path = "FIX_HIT_cilin_utf8CT_zhconv (1).txt"

# 步驟 1: 讀取並將資料依據 (分類字母, 詞數) 進行分組
lines_by_group = defaultdict(list)
total_available_pairs = 0

train_only_lines = []

with open(file_path, "r", encoding="utf-8") as f:
    # 👉 修改：使用 enumerate(f) 取得行號 (line_idx)，方便後續恢復順序
    for line_idx, line in enumerate(f):
        original_line = line.strip()
        if not original_line: continue
        parts = original_line.split()
        if not parts: continue
        
        code_part = parts[0]
        
        # 如果是 # 結尾，連同「行號」一起存起來
        if code_part.endswith('#'): 
            train_only_lines.append((line_idx, original_line))
            continue
        
        word_count = len(parts) - 1
        if word_count >= 2:
            first_letter = code_part[0].upper()
            # 正常資料也連同「行號」一起存起來
            lines_by_group[(first_letter, word_count)].append((line_idx, original_line))
            total_available_pairs += math.comb(word_count, 2)

# 步驟 2: 計算 13000 組的完美抽樣配額 (此段邏輯不變)
target_total_pairs = 13000
ratio = target_total_pairs / total_available_pairs

allocation = {}
current_pairs = 0
fractional_info = []

for (cat, n), lines in lines_by_group.items():
    count = len(lines)
    max_alloc_possible = count // 2
    
    ideal_rows = count * ratio
    allocated_rows = math.floor(ideal_rows)
    allocated_rows = min(allocated_rows, max_alloc_possible)
    
    pairs_per_row = math.comb(n, 2)
    
    allocation[(cat, n)] = allocated_rows
    current_pairs += allocated_rows * pairs_per_row
    
    fractional_rows = ideal_rows - allocated_rows
    expected_pairs_lost = fractional_rows * pairs_per_row
    
    fractional_info.append({
        'key': (cat, n),
        'lost': expected_pairs_lost,
        'max_count': count,
        'max_alloc_possible': max_alloc_possible
    })

fractional_info.sort(key=lambda x: x['lost'], reverse=True)
idx = 0
attempts = 0
while current_pairs < target_total_pairs and attempts < 10000:
    key = fractional_info[idx]['key']
    max_alloc = fractional_info[idx]['max_alloc_possible']
    
    if allocation[key] < max_alloc:
        allocation[key] += 1
        current_pairs += math.comb(key[1], 2)
        
    idx += 1
    if idx == len(fractional_info): 
        idx = 0
    attempts += 1

# 步驟 3: 隨機抽樣並切分 8:1:1
random.seed(42)

# 先用 tuples 暫存 (行號, 內容)
train_data_tuples = []
valid_data_tuples = []
test_data_tuples = []

for (cat, n), lines in lines_by_group.items():
    # 這裡打亂的是 (行號, 內容) 的組合，隨機性不受影響
    random.shuffle(lines)
    alloc_count = allocation[(cat, n)]
    
    valid_slice = lines[0 : alloc_count]
    test_slice = lines[alloc_count : alloc_count * 2]
    train_slice = lines[alloc_count * 2 : ]
    
    valid_data_tuples.extend(valid_slice)
    test_data_tuples.extend(test_slice)
    train_data_tuples.extend(train_slice)

# 將 # 資料加入 Train 暫存列
train_data_tuples.extend(train_only_lines)

# 👉 關鍵修改：依照原始行號 (x[0]) 進行排序，恢復原始順序！
train_data_tuples.sort(key=lambda x: x[0])
valid_data_tuples.sort(key=lambda x: x[0])
test_data_tuples.sort(key=lambda x: x[0])

# 👉 關鍵修改：排序完成後，把純文字部分 (x[1]) 抽出來，丟掉行號
train_data = [x[1] for x in train_data_tuples]
valid_data = [x[1] for x in valid_data_tuples]
test_data = [x[1] for x in test_data_tuples]


# 步驟 4: 驗證與匯出檔案 (此段邏輯不變)
def count_pairs(data_list):
    pairs = 0
    for line in data_list:
        parts = line.split()
        word_count = len(parts) - 1
        if word_count >= 2:
            pairs += math.comb(word_count, 2)
    return pairs

print(f"Valid 配對數: {count_pairs(valid_data)}")
print(f"Test 配對數: {count_pairs(test_data)}")
print(f"Train 配對數: {count_pairs(train_data)}")
print(f"總配對數: {count_pairs(valid_data) + count_pairs(test_data) + count_pairs(train_data)}")

def save_to_file(filename, data_list):
    with open(filename, "w", encoding="utf-8") as f:
        for line in data_list:
            f.write(line + "\n")

save_to_file("dataset_valid.txt", valid_data)
save_to_file("dataset_test.txt", test_data)
save_to_file("dataset_train.txt", train_data)
print("檔案切分匯出完成！")