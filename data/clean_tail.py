import json

def remove_suffix_from_jsonl(input_file, output_file):
    suffixes = ['WR', 'BR', ' B', ' W']
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            text = data.get('text', '')
            
            for suffix in suffixes:
                if text.endswith(suffix):
                    text = text[:-len(suffix)-1]
                    break 
            
            data['text'] = text
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

remove_suffix_from_jsonl('/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/RWKV_GooseGooseGo/data/cleaned_output.jsonl', 'output.jsonl')