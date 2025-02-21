#!/bin/bash
CASE_NAME="40aec5fffa"

root_path="../eval_datasets"

CUDA_VISIBLE_DEVICES=1 python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path} \
        --output_dir ../eval_result \
        --mask_thresh 0.5 \
        --json_folder ${root_path}