import re
import json

def clean_go_dataset(input_file, output_file):
    """
    清洗围棋数据集，只保留字母和空格，确保token之间用单个空格分隔
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                # 只保留字母和空格，不保留数字
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = cleaned_text.strip()
                data['text'] = cleaned_text
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

def clean_go_dataset_advanced(input_file, output_file):
    """
    保留字母、数字、空格和X（虚手标记）
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                
                # 保留字母、数字、空格和X（虚手标记）
                cleaned_text = re.sub(r'[^a-zA-Z0-9\sX]', '', text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = cleaned_text.strip()
                data['text'] = cleaned_text
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

if __name__ == "__main__":
    # 基础清洗版本
    clean_go_dataset('/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/RWKV_GooseGooseGo/data/datasets_final.jsonl', 'cleaned_output.jsonl')
    
    # 高级清洗版本（推荐用于围棋数据集）保留数字
    # clean_go_dataset_advanced('/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/RWKV_GooseGooseGo/data/datasets_final.jsonl', 'cleaned_output.jsonl')