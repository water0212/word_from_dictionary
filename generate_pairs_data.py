import json
import math
import random
import time
from collections import defaultdict
import os
import pandas as pd

# 路徑設定 (請依據您的實際環境確認)
current_dir = os.path.dirname(os.path.abspath(__file__))
pairs_output = os.path.join(current_dir, "pairs_output")
if not os.path.exists(pairs_output):
    os.makedirs(pairs_output)
output_file = os.path.join(pairs_output, "experiment_pairs.json")

# 三個同義詞分割檔 (用來計算數量與定義 Valid/Test)
split_files = {
    'Valid': os.path.join(current_dir, 'divide_dataset_output', 'dataset_valid.txt'),
    'Test': os.path.join(current_dir, 'divide_dataset_output', 'dataset_test.txt'),
    'Train': os.path.join(current_dir, 'divide_dataset_output', 'dataset_train.txt')
}

# 完整詞林檔案 (用來作為 D1~D5 的全局大水池)
full_dict_file = os.path.join(current_dir,"Source", 'FIX_HIT_cilin_utf8_no_empty_poly.txt')

def get_combined_split(s1, s2):
    """
    判斷兩個詞組合後的所屬資料集
    優先度: Valid(3) > Test(2) > Train(1) > None(0)
    若兩者皆小於等於 1 (例如 Train + None, 或 None + None)，則歸入 Train
    """
    priority = {'Valid': 3, 'Test': 2, 'Train': 1, 'None': 0}
    p1 = priority.get(s1, 0)
    p2 = priority.get(s2, 0)
    max_p = max(p1, p2)
    
    if max_p == 3: return 'Valid'
    if max_p == 2: return 'Test'
    return 'Train'

def main():
    # ==========================================
    # 步驟 1: 讀取分割檔，確立同義詞名單與 Split 屬性
    # ==========================================
    code_to_split = {}
    synonym_codes = set() 
    
    print("正在讀取 Train/Valid/Test 分割檔...")
    for split, filename in split_files.items():
        if not os.path.exists(filename):
            print(f"找不到檔案: {filename}")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                code = parts[0][:7]
                code_to_split[code] = split
                synonym_codes.add(code)

    # ==========================================
    # 步驟 2: 讀取完整詞林，建立全局大水池
    # ==========================================
    code_to_words = {}
    prefix_sets = {1: defaultdict(set), 2: defaultdict(set), 
                   3: defaultdict(set), 4: defaultdict(set)}
    all_codes_set = set()
    letter_to_codes = defaultdict(list)

    print("正在讀取完整詞林建立全局水池...")
    with open(full_dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            code = parts[0][:7]
            words = parts[1:]
            
            code_to_words[code] = words
            all_codes_set.add(code)
            letter_to_codes[code[0]].append(code)
            
            # 建立前綴索引
            prefix_sets[1][code[:1]].add(code)
            prefix_sets[2][code[:2]].add(code)
            prefix_sets[3][code[:4]].add(code)
            prefix_sets[4][code[:5]].add(code)
            
            # 如果這個代碼不在分割檔裡 (代表它是 # 或 @)
            # 給予 'None' 屬性，確保它不會污染 Valid 和 Test
            if code not in code_to_split:
                code_to_split[code] = 'None'

    # ==========================================
    # 步驟 3: 產生同義詞配對並計算目標數量
    # ==========================================
    print("正在產生同義詞配對並計算配額...")
    output_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    target_counts = defaultdict(lambda: defaultdict(int))

    # 這裡只針對 synonym_codes (存在於分割檔中的 = 號資料) 產生同義詞
    for code in synonym_codes:
        words = code_to_words.get(code, [])
        if not words:
            continue
        letter = code[0]
        split = code_to_split[code]
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                key = f"{code}, {code}"
                
                counter = 1
                final_key = key
                while final_key in output_data[letter][split]['Synonym']:
                    final_key = f"{key}_{counter}"
                    counter += 1
                    
                output_data[letter][split]['Synonym'][final_key] = [w1, w2]
                target_counts[letter][split] += 1

        # ==========================================
    # 步驟 4: 根據正確的 D1~D5 規則抽取配對 (精準錨點抽樣法 + 進度監控)
    # ==========================================
    def get_c2_candidates(c1, category):
        """正確的 D1~D5 規則 (D1最相似，D5最不相似)"""
        if category == 'D1': return list(prefix_sets[4][c1[:5]] - {c1})
        if category == 'D2': return list(prefix_sets[3][c1[:4]] - prefix_sets[4][c1[:5]])
        if category == 'D3': return list(prefix_sets[2][c1[:2]] - prefix_sets[3][c1[:4]])
        if category == 'D4': return list(prefix_sets[1][c1[:1]] - prefix_sets[2][c1[:2]])
        if category == 'D5': return list(all_codes_set - prefix_sets[1][c1[:1]])
        return []

    print("\n" + "="*50)
    print("🚀 正在從全局水池精準抽取 D1 ~ D5 配對...")
    print("="*50)
    categories = ['D1', 'D2', 'D3', 'D4', 'D5']
    
    codes_by_letter_and_split = defaultdict(lambda: defaultdict(list))
    for c in all_codes_set:
        codes_by_letter_and_split[c[0]][code_to_split[c]].append(c)
    
    for letter, splits in target_counts.items():
        print(f"\n▶️ 開始處理字母分類: [{letter}]")  # 【Log】提示目前處理的大類別
        
        for split, target_n in splits.items():
            if target_n == 0: continue
            
            anchor_codes = codes_by_letter_and_split[letter][split]
            if not anchor_codes:
                print(f"  ⚠️ 警告: {letter} 類完全沒有 {split} 的代碼，無法產生配對。")
                continue
            
            print(f"  👉 處理資料集: {split} (目標配額: {target_n} 組)")  # 【Log】提示目前處理的資料集
            
            for cat in categories:
                sampled_word_pairs = set()
                attempts = 0
                max_attempts = target_n * 50 
                
                start_time = time.time()  # 【Log】開始計時
                
                while len(sampled_word_pairs) < target_n and attempts < max_attempts:
                    attempts += 1
                    
                    # 【Log】每嘗試 10,000 次，回報一次當前進度，證明程式沒當機
                    if attempts % 10000 == 0:
                        print(f"      ⏳ {cat} 努力抽取中... 已獲取: {len(sampled_word_pairs)}/{target_n} (嘗試次數: {attempts})", end='\r')
                    
                    c1 = random.choice(anchor_codes)
                    candidates = get_c2_candidates(c1, cat)
                    if not candidates: continue
                    
                                        # 最佳化寫法：隨機抽盲盒，抽到合格的就停！(極速)
                    c2 = None
                    # 隨機打亂候選名單，最多只檢查 100 個，通常前 3 個就會中了
                    sample_size = min(100, len(candidates))
                    random_sample = random.sample(candidates, sample_size)
                    
                    for temp_c2 in random_sample:
                        if get_combined_split(split, code_to_split[temp_c2]) == split:
                            c2 = temp_c2
                            break  # 找到了就立刻停止檢查！
                            
                    if not c2:
                        continue

                    w1 = random.choice(code_to_words[c1])
                    w2 = random.choice(code_to_words[c2])
                    
                    pair_tuple = tuple(sorted([w1, w2]))
                    if pair_tuple not in sampled_word_pairs:
                        sampled_word_pairs.add(pair_tuple)
                        
                        c_key1, c_key2 = sorted([c1, c2])
                        base_key = f"{c_key1}, {c_key2}"
                        final_key = base_key
                        counter = 1
                        while final_key in output_data[letter][split][cat]:
                            final_key = f"{base_key}_{counter}"
                            counter += 1
                            
                        output_data[letter][split][cat][final_key] = [w1, w2]

                elapsed = time.time() - start_time  # 【Log】結束計時
                
                # 【Log】印出該類別完成的總結報告 (覆蓋掉前面的 \r 進度條)
                print(f"    ✅ {cat} 完成: 取得 {len(sampled_word_pairs)}/{target_n} 組 (耗時 {elapsed:.1f} 秒, 總嘗試 {attempts} 次)".ljust(80))

                # ==========================================
                # 🌟 新增：跨字母借用機制 (當數量不足時，從其他字母補齊)
                # ==========================================
                if len(sampled_word_pairs) < target_n:
                    shortfall = target_n - len(sampled_word_pairs)
                    print(f"    ⚠️ 警告: {letter} 類的 {split} 在 {cat} 缺 {shortfall} 組，準備從其他字母隨機抽取補齊...")
                    
                    # 找出除了自己以外，且在該 split 有代碼的其他字母
                    other_letters = [l for l in codes_by_letter_and_split.keys() if l != letter and len(codes_by_letter_and_split[l][split]) > 0]
                    
                    borrow_attempts = 0
                    max_borrow_attempts = shortfall * 100 # 給予充足的嘗試次數
                    borrow_count = 0
                    
                    while len(sampled_word_pairs) < target_n and borrow_attempts < max_borrow_attempts and other_letters:
                        borrow_attempts += 1
                        
                        # 隨機挑選一個其他的字母作為錨點
                        borrow_letter = random.choice(other_letters)
                        anchor_codes_borrow = codes_by_letter_and_split[borrow_letter][split]
                        
                        c1 = random.choice(anchor_codes_borrow)
                        candidates = get_c2_candidates(c1, cat)
                        if not candidates: continue
                        
                        c2 = None
                        sample_size = min(100, len(candidates))
                        random_sample = random.sample(candidates, sample_size)
                        
                        for temp_c2 in random_sample:
                            if get_combined_split(split, code_to_split[temp_c2]) == split:
                                c2 = temp_c2
                                break
                                
                        if not c2: continue
                            
                        w1 = random.choice(code_to_words[c1])
                        w2 = random.choice(code_to_words[c2])
                        
                        pair_tuple = tuple(sorted([w1, w2]))
                        
                        # 確保這組詞沒有被抽過
                        if pair_tuple not in sampled_word_pairs:
                            sampled_word_pairs.add(pair_tuple)
                            borrow_count += 1
                            
                            c_key1, c_key2 = sorted([c1, c2])
                            base_key = f"{c_key1}, {c_key2}"
                            final_key = base_key
                            counter = 1
                            
                            # 寫入原本字母的 output_data 中 (假裝是它自己產生的)
                            while final_key in output_data[letter][split][cat]:
                                final_key = f"{base_key}_{counter}"
                                counter += 1
                                
                            output_data[letter][split][cat][final_key] = [w1, w2]
                            
                    if len(sampled_word_pairs) < target_n:
                        print(f"    ❌ 跨字母借用後仍不足: 最終取得 {len(sampled_word_pairs)}/{target_n} 組。")
                    else:
                        print(f"    🌟 跨字母借用成功！已補齊 {borrow_count} 組，達到目標 {target_n} 組。")


     # ==========================================
    # 【修改】步驟 4.5: 印出並匯出各類別實際抽取數量總結表 (Excel)
    # ==========================================
    print("\n" + "="*85)
    print("📊 實際抽取數量總結 (供您與理論極限值核對)")
    print("="*85)
    
    # 設定終端機顯示的表格標題
    header = f"{'字母':<4} | {'類別':<4} | {'Train (實際/目標)':<18} | {'Valid (實際/目標)':<18} | {'Test (實際/目標)':<18} | {'總計產出':<10}"
    print(header)
    print("-" * 85)
    
    # 準備用來存入 Excel 的資料列表
    summary_data = []

    for letter in sorted(target_counts.keys()):
        for cat in categories:
            # 取得目標數量
            train_tgt = target_counts[letter]['Train']
            valid_tgt = target_counts[letter]['Valid']
            test_tgt = target_counts[letter]['Test']
            
            # 取得實際成功抽取的數量
            train_act = len(output_data[letter]['Train'][cat])
            valid_act = len(output_data[letter]['Valid'][cat])
            test_act = len(output_data[letter]['Test'][cat])
            
            # 計算該字母在該類別的總產出
            total_act = train_act + valid_act + test_act
            
            # 如果該字母有目標數量才處理
            if train_tgt + valid_tgt + test_tgt > 0:
                train_str = f"{train_act}/{train_tgt}"
                valid_str = f"{valid_act}/{valid_tgt}"
                test_str = f"{test_act}/{test_tgt}"
                
                # 印出格式化的一行數據到終端機
                print(f"{letter:<4} | {cat:<4} | {train_str:<18} | {valid_str:<18} | {test_str:<18} | {total_act:<10}")
                
                # 將數據加入列表，準備匯出 Excel
                # 這裡將「實際」與「目標」拆成獨立欄位，方便您在 Excel 中做後續的加總或圖表分析
                summary_data.append({
                    '字母': letter,
                    '類別': cat,
                    'Train 實際產出': train_act,
                    'Train 目標配額': train_tgt,
                    'Valid 實際產出': valid_act,
                    'Valid 目標配額': valid_tgt,
                    'Test 實際產出': test_act,
                    'Test 目標配額': test_tgt,
                    '總計產出': total_act
                })
        
        # 每個字母印完後加一條分隔線
        print("-" * 85)
    print("\n")

    # 將收集到的資料轉換為 DataFrame 並匯出成 Excel
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_excel_path = os.path.join(pairs_output, "summary_table.xlsx")
        df_summary.to_excel(summary_excel_path, index=False)
        print(f"📈 總結表已成功匯出至 Excel：【{summary_excel_path}】")
    # ==========================================
    # 步驟 5: 匯出 JSON
    # ==========================================
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 處理完成！結果已儲存至 {output_file}")

if __name__ == "__main__":
    main()
