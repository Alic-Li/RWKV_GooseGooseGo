########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# Modified for Go Game Inference
########################################################################################################

print("RWKV GooseGooseGo Go Game Inference Model")

import os, copy, types, sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_V7_ON"] = "1" # enable this for rwkv-7 models
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # !!! '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries !!!

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()
args.strategy = "cuda fp16"  # use CUDA, fp16
args.MODEL_NAME = "out/L24-D384-x070/rwkv-6"

STATE_NAME = None # use vanilla zero initial state?

GEN_TEMP = 1.6
GEN_TOP_P = 0.8
GEN_alpha_presence = 0.5
GEN_alpha_frequency = 0.5
GEN_penalty_decay = 0.996

CHUNK_LEN = 16  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer = TRIE_TOKENIZER("data/tokenizer/rwkv_Goose_Go_vocab.txt")
model_tokens = []
model_state = None

if STATE_NAME != None: # load custom state
    args = model.args
    state_raw = torch.load(STATE_NAME + '.pth')
    state_init = [None for i in range(args.n_layer * 3)]
    for i in range(args.n_layer):
        dd = model.strategy[i]
        dev = dd.device
        atype = dd.atype    
        state_init[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
        state_init[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    model_state = copy.deepcopy(state_init)

def run_rnn(ctx):
    global model_tokens, model_state

    tokens = tokenizer.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f"### model ###\n{model_tokens}\n[{tokenizer.decode(model_tokens)}]")  # debug
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out

def predict_go_move(input_sequence):
    global model_tokens, model_state
    
    # Reset state for each prediction
    model_tokens = []
    model_state = None
    
    occurrence = {}
    out_tokens = []
    out_last = 0
    
    # Run the model
    if not input_sequence:
        # If input is empty (first move), initialize 'out' with a space token to start the RNN state
        out, model_state = model.forward([tokenizer.encode(' ')[0]], None)
    else:
        out = run_rnn(input_sequence)

    # print("Predicted move: ", end="")

    stop_token_id = tokenizer.encode(" ")[0] if tokenizer.encode(" ") else 32  # ASCII for space
    
    for i in range(6):  # Limit to 100 tokens to prevent infinite loop
        # Apply repetition penalty
        for n in occurrence:
            out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency
        out[0] -= 1e10  # disable END_OF_TEXT

        token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)

        out, model_state = model.forward([token], model_state)
        model_tokens += [token]
        out_tokens += [token]

        # Update occurrence for repetition penalty
        for xxx in occurrence:
            occurrence[xxx] *= GEN_penalty_decay
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

        # Decode and print tokens
        tmp = tokenizer.decode(out_tokens[out_last:])
        if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
            print(tmp, end="", flush=True)
            out_last = i + 1

        # Stop when we encounter a space (move separator) AND we have at least 2 tokens
        if token == stop_token_id and len(out_tokens) >= 2:
            print("", end=" ", flush=True)
            break
            
        # Also break if we have a reasonable move prediction (at least 2 chars and ends with space)
        decoded = tokenizer.decode(out_tokens)
        if len(decoded.strip()) >= 2 and decoded.strip()[-1] == ' ' and len(out_tokens) >= 2:
            break
    
    return tokenizer.decode(out_tokens).strip()

def infinite_prediction():
    print("Go Game Infinite Prediction Mode")
    initial_input = input("Enter initial move sequence: ").strip()
    
    if not initial_input:
        print("No initial input provided. Exiting infinite prediction mode.")
        return
    
    current_context = initial_input
    print(f"Starting with: {current_context}")
    
    try:
        while True:
            predicted_move = predict_go_move(current_context)
            
            if predicted_move:
                predicted_move = predicted_move.lstrip()
                
                if predicted_move:  # 确保移除空格后还有内容
                    current_context = predicted_move
                    # print(f"Context updated: {current_context}")
                else:
                    print("No valid move predicted (only whitespace). Stopping infinite prediction.")
                    break
            else:
                print("No move predicted. Stopping infinite prediction.")
                break
            
    except KeyboardInterrupt:
        print("\n\nInfinite prediction interrupted by user.")
    except Exception as e:
        print(f"\nError during infinite prediction: {e}")
    
    print("Exited infinite prediction mode.")


if __name__ == "__main__":
    infinite_prediction()
    