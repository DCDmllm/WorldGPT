#!/bin/bash
export NCCL_P2P_DISABLE=1
deepspeed --master_addr 127.0.0.1 --master_port 28459 train.py \
    --cfg_path config/train.yaml \
    --save_path ckpt/worldgpt \
    --load_path /path/to/worldgpt-languagebind-checkpoint