#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=bert_345M
OUTPUT_DIR=${base_dir}/bert_model_outputs
DATA_DIR=/fsx/datasets

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/configs/bert_large.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 10 \
--lr_schedule "LP" \
--lr_offset 0.0 \
--save_ckpt_interval 50000 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/configs/deepspeed_bert_config.json \
--data_path_prefix ${DATA_DIR} \
--log_file ${JOB_NAME}.log
