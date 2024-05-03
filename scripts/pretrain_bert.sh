#!/bin/bash
warmup_ratio=0.01
exit_steps=20000
# 128 256 384 512 640 768 896 1024 2048 3072
# 5e-4 1e-3 5e-3 1e-2 3e-2 5e-2 7e-2 1e-1
for base_lr in 5e-4 1e-3 5e-3 1e-2 3e-2 5e-2 7e-2 1e-1
do
    for hp_tune_actual_width in 128 256 384 512 640 768 896 1024
    do
        python pretrain_bert.py \
        --base_lr ${base_lr} \
        --hp_tune_actual_width ${hp_tune_actual_width} \
        --tokenizer_path bert-base-uncased \
        --class_name bert \
        --use_mup
        sleep 10
    done
done
