from datasets import load_dataset
import json

ds_go_pgn_string_leela_zero = load_dataset("kenhktsui/go_pgn_string_leela_zero", download_mode="force_redownload")
ds_go_pgn_string_v2 = load_dataset("kenhktsui/go_pgn_string_v2", download_mode="force_redownload")

def save_dataset_to_jsonl(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

for split_name in ds_go_pgn_string_leela_zero.keys():
    save_dataset_to_jsonl(
        ds_go_pgn_string_leela_zero[split_name], 
        f"go_pgn_string_leela_zero_{split_name}.jsonl"
    )

for split_name in ds_go_pgn_string_v2.keys():
    save_dataset_to_jsonl(
        ds_go_pgn_string_v2[split_name], 
        f"go_pgn_string_v2_{split_name}.jsonl"
    )

print("数据集已保存为JSONL格式")