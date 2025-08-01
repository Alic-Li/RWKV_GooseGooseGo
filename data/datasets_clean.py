import json
import re

def transform_jsonl_dataset(input_file, output_file):
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            text = data['text']
            
            # 去除数字和点号
            text = re.sub(r'[\d\.]', '', text)
            
            # 根据规则转换文本
            # 原始文本格式是 "1. >Pc Cp 2. >Cd Qp 3. >Eq Dn 4. >Oq Pn 5. >Qf"
            # 转换为 "PcCp CdQp EqDn OqPn Qf"
            transformed_text = text.replace('>', '')
            transformed_text = transformed_text.replace('  ', ' ')  # 处理多个空格
            transformed_text = transformed_text.strip()  # 去除首尾空格
            
            # 按空格分割并重新组合成每组4个字符的形式
            parts = transformed_text.split(' ')
            # 过滤掉空字符串
            parts = [part for part in parts if part]
            
            # 将相邻的两个部分组合在一起，每组之间用空格分隔
            transformed_text = ' '.join([
                ''.join(parts[i:i+2]) for i in range(0, len(parts), 2)
            ])
            
            # 更新数据并写入文件
            data['text'] = transformed_text
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

transform_jsonl_dataset('/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/datasets_final.jsonl', 'dataset.jsonl')