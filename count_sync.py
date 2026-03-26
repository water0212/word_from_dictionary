import pandas as pd
from collections import defaultdict

file_path = "FIX_HIT_cilin_utf8CT_zhconv.txt"
stats = defaultdict(lambda: defaultdict(int))

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if not parts:
            continue
            
        code_part = parts[0]
        
        # 條件1: 編號以 '#' 結尾的不統計
        if code_part.endswith('#'):
            continue
            
        # 計算這行的詞數
        word_count = len(parts) - 1
        
        # 條件2: 如果這行是 0 個詞，不統計
        if word_count == 0:
            continue
            
        first_letter = code_part[0].upper()
        stats[first_letter][word_count] += 1

# --- 接著整理成 DataFrame 並輸出為 EXCEL 檔案 ---
records = []
for category, counts in stats.items():
    record = {'分類字母': category}
    record.update(counts)
    records.append(record)

df = pd.DataFrame(records)
df = df.fillna(0)

count_cols = [col for col in df.columns if isinstance(col, int)]
count_cols.sort()
final_cols = ['分類字母'] + count_cols
df = df[final_cols]
df = df.sort_values('分類字母')

for col in count_cols:
    df[col] = df[col].astype(int)
    
rename_dict = {col: f"{col}個詞" for col in count_cols}
df = df.rename(columns=rename_dict)

excel_path = "filtered_category_word_count_distribution.xlsx"
df.to_excel(excel_path, index=False)