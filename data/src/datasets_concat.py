#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

def extract_text_from_jsonl(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        texts.append({'text': data['text']})
                except json.JSONDecodeError:
                    print(f"警告: 无法解析 {file_path} 中的一行: {line}")
                    continue
    return texts

def save_texts_to_jsonl(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='从JSONL数据集中提取text键并合并乱序')
    parser.add_argument('input_files', nargs='+', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', default='merged_dataset.jsonl', help='输出文件名')
    args = parser.parse_args()
    all_texts = []
    
    for file_path in args.input_files:
        if Path(file_path).exists():
            print(f"处理文件: {file_path}")
            texts = extract_text_from_jsonl(file_path)
            all_texts.extend(texts)
            print(f"从 {file_path} 中提取了 {len(texts)} 条记录")
        else:
            print(f"警告: 文件 {file_path} 不存在，跳过")
    
    print(f"总共收集到 {len(all_texts)} 条记录，正在进行乱序...")
    random.shuffle(all_texts)
    
    save_texts_to_jsonl(all_texts, args.output)
    print(f"数据已保存到 {args.output}")

if __name__ == '__main__':
    main()