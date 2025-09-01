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
args.MODEL_NAME = "rwkv-2981"

STATE_NAME = None # use vanilla zero initial state? 

GEN_TEMP = 1.5
GEN_TOP_P = 0

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer = TRIE_TOKENIZER("data/tokenizer/rwkv_Goose_Go_vocab.txt")
model_state = None
init_state = None

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
    init_state = copy.deepcopy(state_init)

def reset_model_state():
    """Resets the RNN state. Call this before starting a new game."""
    global model_state
    model_state = copy.deepcopy(init_state) if init_state is not None else None
    print("Model state has been reset.")

def predict_go_move(input_move_notation):
    """
    Predicts the next move based on the previous single move.
    Manages the RNN state internally.
    """
    global model_state
    
    # If input is None (start of the game for white player), we need a starting token.
    # We use token 0, which is a common practice for starting a sequence.
    if input_move_notation is None:
        tokens = [0]
    else:
        tokens = tokenizer.encode(input_move_notation)
        if not tokens:
            print(f"Warning: Could not encode move '{input_move_notation}'. Using a default token.")
            tokens = [0] # Fallback if encoding fails

    # Forward pass to update state and get logits
    out, model_state = model.forward(tokens, model_state)
    
    # Sample the next token
    token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)
    
    return tokenizer.decode([token])

def infinite_prediction():
    print("Go Game Infinite Prediction Mode")
    reset_model_state()
    initial_input = input("Enter initial move sequence (or leave blank to start): ").strip()
    
    if initial_input:
        # Prime the model with the initial sequence
        print(f"Priming with: {initial_input}")
        predict_go_move(initial_input)
    
    last_move = initial_input if initial_input else None
    
    try:
        while True:
            predicted_move = predict_go_move(last_move)
            
            if predicted_move == '\ufffd':
                print("\nEnd of sequence (special character detected)")
                break
                
            print(f"{predicted_move}", end='', flush=True) 
            last_move = predicted_move
            
    except KeyboardInterrupt:
        print("\n\nInfinite prediction interrupted by user.")
    except Exception as e:
        print(f"\nError during infinite prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("Exited infinite prediction mode.")

def infer_from_sequence(input_sequence):

    global model_state
    
    reset_model_state()
    print(input_sequence)
    
    tokens = tokenizer.encode(input_sequence)
    
    if not tokens:
        print("Warning: Empty input sequence. Using default token.")
        tokens = [0]
    
    logits = None
    current_state = model_state
    
    for i, token in enumerate(tokens):
        logits, current_state = model.forward([token], current_state)
    
    token = pipeline.sample_logits(logits, temperature=GEN_TEMP, top_p=GEN_TOP_P)
    return tokenizer.decode([token])


if __name__ == "__main__":
    # infinite_prediction()
    input_sequence = "###################\n##W################\nWBB#W##############\n#######B#####W#B###\nB##################\n##B##B#############\n###################\n###################\n###################\n###W###########W###\n###################\n############W######\n###################\n##B################\nBB#################\n###W#B#B#W#####W###\n#BB################\n###WWW###W#########\n###################\nBlack"
    next_move = infer_from_sequence(input_sequence)
    print(f"Predicted next move: {next_move}")