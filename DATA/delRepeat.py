import json
import os
import re
from typing import List, TypedDict

BASE_DIR = os.path.dirname(__file__)
INPUT_TXT = os.path.join(BASE_DIR, "HIT_cilin_utf8CT_zhconv.txt")
FIX_TXT = os.path.join(BASE_DIR, "repeat_word.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "FIX_HIT_cilin_utf8CT_zhconv.txt")


def parse_hit_cilin(txt_path, fix_path):
    result = ""
    num = {}
    txt_pattern = re.compile(r"^(\S+?)([=#@])\s*(.*)$")
    fix_pattern = re.compile(r'^\s*(?P<word>[^:]+?)\s*:\s*(?P<codes>[A-Za-z]{2}\d{2}[A-Za-z]\d{2}[=#@](?:\s*,\s*[A-Za-z]{2}\d{2}[A-Za-z]\d{2}[=#@])*)\s*$')
    data = {}
    # Read the fix data
    with open(fix_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = fix_pattern.match(line)
            if m:
                word = m.group("word")
                codes = [code.strip() for code in m.group("codes").split(",")]
                data[word] = codes

    with open(txt_path, "r", encoding="utf-8") as t:
        for line_no, raw_line in enumerate(t, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = txt_pattern.match(line)
            if not match:
                continue

            code, relation, words_part = match.groups()
            str= code + relation
            for w in words_part.split():
                if w in data and len(data[w]) == 1:
                    str += " " + w
            str = str.strip()
            result += str + "\n"
    return result


def save_txt(data, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(data)


if __name__ == "__main__":
    data = parse_hit_cilin(INPUT_TXT, FIX_TXT)
    save_txt(data, OUTPUT_JSON)
    print(f"完成：共 {len(data)} 筆，已輸出到 {OUTPUT_JSON}")