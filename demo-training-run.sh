#!/bin/bash
################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
################################################################################
#
MODEL_TYPE="x070" # x060 => rwkv-6.0
#
N_LAYER="24"
N_EMBD="384"
#
CTX_LEN="1024" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
M_BSZ="6" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
LR_INIT="4e-4"
LR_FINAL="4e-6"
GRAD_CP=0 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=1 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
# --my_exit_tokens 875602734 --magic_prime 855059 --ctx_len 1024
MY_EXIT_TOKENS="875602734"
MAGIC_PRIME="855059"
# DATA_FILE="data/pretrain_hq"
DATA_FILE="data/dataset_cleaned"
DATA_TYPE="binidx"
#
python train.py \
 --wandb "RWKV_Goose_Go" \
 --accelerator gpu \
 --adam_eps 1e-18 \
 --beta1 0.9 \
 --beta2 0.99 \
 --ctx_len $CTX_LEN \
 --tokenizer " " \
 --data_file $DATA_FILE \
 --data_type $DATA_TYPE \
 --devices $GPU_PER_NODE \
 --ds_bucket_mb $DS_BUCKET_MB \
 --enable_progress_bar True \
 --epoch_begin 0 \
 --epoch_count 999999 \
 --epoch_save $EPOCH_SAVE \
 --grad_cp $GRAD_CP \
 --head_size 64 \
 --load_model "0" \
 --lr_final $LR_FINAL \
 --lr_init $LR_INIT \
 --magic_prime $MAGIC_PRIME \
 --micro_bsz $M_BSZ \
 --my_exit_tokens $MY_EXIT_TOKENS \
 --my_testing $MODEL_TYPE \
 --n_embd $N_EMBD \
 --n_layer $N_LAYER \
 --num_nodes $N_NODE \
 --precision bf16 \
 --proj_dir $PROJ_DIR \
 --strategy deepspeed_stage_2 \
 --train_stage 3 \
 --vocab_size 364 \
 --warmup_steps 10 \
 --weight_decay 0.001 \
