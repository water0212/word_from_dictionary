import pandas as pd
from collections import defaultdict
import os

# ==========================================
# ⚙️ 設定區
# ==========================================
INPUT_FILE = 'FIX_HIT_cilin_utf8_no_empty_poly.txt'  # 您的同義詞林檔案
OUTPUT_FILE = 'Cilin_D1_to_D5_Statistics.xlsx'       # 輸出的 Excel 檔名
current_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(current_dir, INPUT_FILE)
OUTPUT_FILE = os.path.join(current_dir, OUTPUT_FILE)
# ==========================================
# 🚀 執行區
# ==========================================
def calculate_statistics():
    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ 找不到檔案 {INPUT_FILE}，請確認路徑！")
        return

    print(f"📂 正在讀取並解析 {INPUT_FILE} ...")
    code_to_words = defaultdict(list)
    
    # 1. 讀取檔案並建立 code -> words 的對應
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            if len(parts) < 2: continue
            
            raw_code = parts[0]
            code = raw_code[:7] # 只取前 7 碼 (例如 Aa01A01，忽略最後的 = 或 #)
            words = parts[1:]
            
            # 將單字加入該代碼中 (用 set 去重避免重複計算)
            code_to_words[code].extend(words)
            code_to_words[code] = list(set(code_to_words[code]))

    # 準備統計字典
    stats = defaultdict(lambda: {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0})
    
    # 預先計算每個字母的總詞數 (用於快速計算 D5)
    letter_word_count = defaultdict(int)
    for code, words in code_to_words.items():
        letter = code[0]
        letter_word_count[letter] += len(words)
        
    total_words_all = sum(letter_word_count.values())
    
    # 取得所有出現過的大寫字母
    letters = sorted(list(letter_word_count.keys()))
    all_codes = sorted(list(code_to_words.keys()))
    
    print("⏳ 正在計算 D1 ~ D4 的配對數量 (這可能需要幾秒鐘)...")
    
    # 2. 計算 D1 ~ D4
    for letter in letters:
        # 篩選出該字母開頭的所有代碼
        codes_in_letter = [c for c in all_codes if c.startswith(letter)]
        n = len(codes_in_letter)
        
        # 兩兩代碼進行配對 (C 取 2 的概念)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = codes_in_letter[i]
                c2 = codes_in_letter[j]
                
                # 這兩個代碼能產生的配對總數 = 詞數相乘
                pairs_count = len(code_to_words[c1]) * len(code_to_words[c2])
                
                # 判斷關係並歸類
                if c1[:5] == c2[:5]:
                    stats[letter]['D1'] += pairs_count
                elif c1[:4] == c2[:4]:
                    stats[letter]['D2'] += pairs_count
                elif c1[:2] == c2[:2]:
                    stats[letter]['D3'] += pairs_count
                else:
                    stats[letter]['D4'] += pairs_count

    print("⏳ 正在計算 D5 的配對數量...")
    
    # 3. 計算 D5 (跨字母配對)
    for letter in letters:
        # 該字母的詞數 × (所有字母總詞數 - 該字母詞數)
        # 代表這個字母裡面的所有詞，去跟「其他所有字母」的詞配對
        stats[letter]['D5'] = letter_word_count[letter] * (total_words_all - letter_word_count[letter])

    # 4. 整理成 DataFrame 準備輸出
    print("📊 正在產生 Excel 報表...")
    data = []
    for letter in letters:
        data.append({
            '大寫字母': letter,
            '該字母總詞數': letter_word_count[letter],
            'D1 (前5碼相同)': stats[letter]['D1'],
            'D2 (前4碼相同)': stats[letter]['D2'],
            'D3 (前2碼相同)': stats[letter]['D3'],
            'D4 (前1碼相同)': stats[letter]['D4'],
            'D5 (跨大類配對)': stats[letter]['D5']
        })
        
    df = pd.DataFrame(data)
    
    # 加入總計行
    total_row = {
        '大寫字母': '總計 (Total)',
        '該字母總詞數': df['該字母總詞數'].sum(),
        'D1 (前5碼相同)': df['D1 (前5碼相同)'].sum(),
        'D2 (前4碼相同)': df['D2 (前4碼相同)'].sum(),
        'D3 (前2碼相同)': df['D3 (前2碼相同)'].sum(),
        'D4 (前1碼相同)': df['D4 (前1碼相同)'].sum(),
        'D5 (跨大類配對)': df['D5 (跨大類配對)'].sum() // 2  # D5 總計要除以 2，避免 A配B 和 B配A 重複計算
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # 輸出 Excel
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ 計算完成！結果已儲存至：【{OUTPUT_FILE}】")

if __name__ == '__main__':
    calculate_statistics()
