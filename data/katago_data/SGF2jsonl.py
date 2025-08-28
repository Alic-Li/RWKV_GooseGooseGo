import re
import json
import os

def sgf_to_json_text(sgf_path):
    with open(sgf_path, 'r', encoding='utf-8') as f:
        sgf_content = f.read()

    # 匹配所有落子坐标（B[xx] 或 W[xx]）
    moves = re.findall(r';[BW]\[([a-zA-Z]{2})\]', sgf_content)

    # 每个坐标首字母大写
    moves = [m[0].upper() + m[1] for m in moves]

    # 每两个坐标合并为一个字符串
    paired_moves = []
    i = 0
    while i < len(moves):
        if i + 1 < len(moves):
            paired_moves.append(moves[i] + moves[i + 1])
            i += 2
        else:
            paired_moves.append(moves[i] + "X")
            i += 1

    # 拼接为最终字符串
    move_str = ' '.join(paired_moves)
    return {"text": move_str}

def process_folder(folder_path, output_jsonl_path):
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.sgf'):
                    sgf_path = os.path.join(root, file)
                    try:
                        result = sgf_to_json_text(sgf_path)
                        out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                        print(f"✅ 处理完成：{sgf_path}")
                    except Exception as e:
                        print(f"❌ 错误处理文件 {sgf_path}: {e}")

# 示例用法
input_folder = "data/katago_data/"
output_jsonl = "katago_output.jsonl"
process_folder(input_folder, output_jsonl)
