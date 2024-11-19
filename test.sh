#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST

eval "$(conda shell.bash hook)"
conda activate refii

EXP_NAME="diff_fe_sigma_2_fe"
CKPT="/data1/yipeng_wei/data/exp/diff_fe_sigma_2/ckpt/model_epoch_best.pth"
DATASETS_TEST="diff_fe"
python /data1/yipeng_wei/ReFii/test.py --gpus 2 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST 2>&1 | tee test.txt
