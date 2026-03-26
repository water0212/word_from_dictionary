import json
import os
import re
from typing import Dict,TypedDict

BASE_DIR = os.path.dirname(__file__)
INPUT_TXT = os.path.join(BASE_DIR, "FIX_HIT_cilin_utf8CT_zhconv.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "zhconv_result.json")

class GroupStats(TypedDict):
    lines: int
    total_num: int
    pair_num: list[int]  # 長度固定 6


def parse_hit_cilin(txt_path):
    result: Dict[str, GroupStats] = {}
    line_pattern = re.compile(r"^(\S+?)([=#@])\s*(.*)$")
    code_pattern = re.compile(r"^([A-Za-z])([A-Za-z])(\d{2})([A-Za-z])(\d{2})$")
    num: Dict[str, int] = {}

    def ensure_group(group_key: str) -> None:
        if group_key not in result:
            result[group_key] = {"lines": 0, "total_num": 0, "pair_num": [0, 0, 0, 0, 0, 0]}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = line_pattern.match(line)
            if not match:
                continue

            code, relation, words_part = match.groups()
            words = [w for w in words_part.split() if w]
            num[code] = len(words)
            if relation != "#":
                group_key = code[0]
                pair_count = len(words) * (len(words) - 1) // 2
                ensure_group(group_key)
                result[group_key]["lines"] += 1
                result[group_key]["total_num"] += len(words)
                result[group_key]["pair_num"][0] += pair_count

    for key1, value1 in num.items():
        m1 = code_pattern.match(key1)
        if not m1:
            continue
        r11, r12, r13, r14, r15 = m1.groups()
        ensure_group(r11)
        for key2, value2 in num.items():
            m2 = code_pattern.match(key2)
            if not m2:
                continue
            r21, r22, r23, r24, r25 = m2.groups()
            ensure_group(r21)

            score = value1 * value2
            if r11 == r21:
                if r12 == r22:
                    if r13 == r23:
                        if r14 == r24:
                            if r15 == r25:
                                continue
                            result[r11]["pair_num"][1] += score
                        else:
                            result[r11]["pair_num"][2] += score
                    else:
                        result[r11]["pair_num"][3] += score
                else:
                    result[r11]["pair_num"][4] += score
            else:
                result[r11]["pair_num"][5] += score
                result[r21]["pair_num"][5] += score

    for code, values in result.items():
        print(f"{code}: {values}")
    return result


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data = parse_hit_cilin(INPUT_TXT)
    save_json(data, OUTPUT_JSON)
    print(f"完成：共 {len(data)} 筆，已輸出到 {OUTPUT_JSON}")