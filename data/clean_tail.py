import json

def remove_text_after_last_space(input_file, output_file):
    """
    去除jsonl文件中每个text字段最后一个空格及其后的文本
    例如: "KeGh GgOq B+R" -> "KeGh GgOq"
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            text = data.get('text', '')
            
            # 找到最后一个空格的位置并截取之前的部分
            if ' ' in text:
                last_space_index = text.rfind(' ')
                text = text[:last_space_index]
            
            data['text'] = text
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例
remove_text_after_last_space('/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/RWKV_GooseGooseGo/dataset.jsonl', '/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/RWKV_GooseGooseGo/dataset_cleaned.jsonl')