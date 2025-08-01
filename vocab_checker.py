
import json
import sys

def main():
    vocab_file = 'data/tokenizer/rwkv_Goose_Go_vocab.txt'
    jsonl_file = 'data/output.jsonl'

    # Read the vocabulary into a set
    vocab = set()
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    try:
                        # The line format is like: 30004 "的" 1
                        # We need to extract the character in the quotes
                        token_str = eval(parts[1].rsplit(' ', 1)[0])
                        vocab.update(token_str)
                    except:
                        # Handle potential parsing errors if a line is not in the expected format
                        pass
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_file}", file=sys.stderr)
        sys.exit(1)

    # Read the jsonl file and find characters not in vocab
    missing_chars = set()
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Assuming the jsonl has a "text" and/or "title" field
                    data = json.loads(line)
                    text = data.get("text", "") + data.get("title", "")
                    for char in text:
                        if char not in vocab:
                            missing_chars.add(char)
                except (json.JSONDecodeError, AttributeError):
                    # If not a valid json or no text/title, check the whole line
                    for char in line:
                        if char not in vocab:
                            missing_chars.add(char)
    except FileNotFoundError:
        print(f"Error: Data file not found at {jsonl_file}", file=sys.stderr)
        sys.exit(1)

    # Print the results
    if missing_chars:
        sorted_missing = sorted(list(missing_chars))
        print(f"不在词汇表中的字符: {' '.join(sorted_missing)}")
    else:
        print("所有字符都在词汇表中。")

if __name__ == "__main__":
    main()
