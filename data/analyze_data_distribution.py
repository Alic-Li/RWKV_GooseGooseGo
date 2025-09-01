
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Assuming the tokenizer script is in the 'tokenizer' subdirectory.
# We need to adjust the path to import it correctly.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tokenizer'))
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# --- Configuration ---
JSONL_FILE = 'val_data.jsonl'
TOKENIZER_VOCAB_FILE = 'tokenizer/rwkv_Goose_Go_vocab.txt'
OUTPUT_IMAGE_FILE = 'token_distribution.png'
BINS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

def analyze_distribution():
    """
    Analyzes the token count distribution of a jsonl file and generates a bar chart.
    """
    # --- 1. Initialize Tokenizer ---
    print(f"Initializing tokenizer with vocab from: {TOKENIZER_VOCAB_FILE}")
    if not os.path.exists(TOKENIZER_VOCAB_FILE):
        print(f"Error: Tokenizer vocab file not found at '{TOKENIZER_VOCAB_FILE}'")
        return
    tokenizer = TRIE_TOKENIZER(TOKENIZER_VOCAB_FILE)

    # --- 2. Initialize Bins for Counting ---
    bin_counts = {f"0-{BINS[0]}": 0}
    for i in range(len(BINS) - 1):
        bin_counts[f"{BINS[i] + 1}-{BINS[i+1]}"] = 0
    bin_counts[f">{BINS[-1]}"] = 0
    
    # Add a counter for entries that couldn't be processed
    error_count = 0

    # --- 3. Process the JSONL File ---
    print(f"Analyzing token distribution for: {JSONL_FILE}")
    if not os.path.exists(JSONL_FILE):
        print(f"Error: Input file not found at '{JSONL_FILE}'")
        return

    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        # Use tqdm for a progress bar
        for line in tqdm(f, desc="Processing lines"):
            try:
                # Load the JSON object from the line
                data = json.loads(line)
                text = data.get('text')

                if text and isinstance(text, str):
                    # Encode the text to get token count
                    token_count = len(tokenizer.encode(text))

                    # Find the correct bin and increment its count
                    if token_count <= BINS[0]:
                        bin_counts[f"0-{BINS[0]}"] += 1
                    elif token_count > BINS[-1]:
                        bin_counts[f">{BINS[-1]}"] += 1
                    else:
                        for i in range(len(BINS) - 1):
                            if BINS[i] < token_count <= BINS[i+1]:
                                bin_counts[f"{BINS[i] + 1}-{BINS[i+1]}"] += 1
                                break
                else:
                    error_count += 1
            except (json.JSONDecodeError, AttributeError):
                error_count += 1
                continue
    
    print("Analysis complete.")
    if error_count > 0:
        print(f"Warning: Skipped {error_count} lines due to parsing errors or missing 'text' field.")

    # --- 4. Generate the Bar Chart ---
    print("Generating bar chart...")
    
    labels = list(bin_counts.keys())
    counts = list(bin_counts.values())

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, counts, color='skyblue')
    
    # Add counts on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.xlabel('Token Count Bins')
    plt.ylabel('Number of Samples')
    plt.title(f'Token Count Distribution in {JSONL_FILE}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Adjust layout to make room for rotated labels

    # Save the plot to a file
    plt.savefig(OUTPUT_IMAGE_FILE)
    print(f"Chart saved to {OUTPUT_IMAGE_FILE}")

if __name__ == '__main__':
    analyze_distribution()
