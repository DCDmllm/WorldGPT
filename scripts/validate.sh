#!/bin/bash
export NCCL_P2P_DISABLE=1
deepspeed --master_addr 127.0.0.1 --master_port 28459 train.py \
    --eval_only --cfg_path config/validate.yaml \
    --log_path log/validate \
    --load_path /path/to/worldgpt-languagebind-checkpoint