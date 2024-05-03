#!/bin/bash
warmup_ratio=0.01
exit_steps=20000
for base_lr in 5e-4 1e-3 5e-3 1e-2 3e-2 5e-2 7e-2 1e-1
do
    for hp_tune_actual_width in 128 256 384 512 640 768 896 1024
    do
        python pretrain_t5.py \
        --base_lr ${base_lr} \
        --hp_tune_actual_width ${hp_tune_actual_width} \
        --tokenizer_path google/t5-v1_1-base \
        --class_name t5 \
        --use_mup 
        sleep 10
    done
done