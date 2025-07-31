# -*- coding: utf-8 -*-
import json
import sys

def convert_json_to_txt(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f_json:
            data = json.load(f_json)
        with open(output_path, 'w', encoding='utf-8') as f_txt:
            for key, value in data.items():
                new_line = f"{value+1} '{key}' 1\n"
                f_txt.write(new_line)
        print(f"转换成功！已将 {input_path} 转换为 {output_path}")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 {input_path}。请检查文件格式是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python json_to_txt.py <输入json文件路径> <输出txt文件路径>")
        print("例如: python json_to_txt.py input.json output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_json_to_txt(input_file, output_file)
