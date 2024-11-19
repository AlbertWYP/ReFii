#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate refii
EXP_NAME="diff_fe_sigma_2"
DATASETS="diff_fe"
DATASETS_TEST="diff_fe"
#python train.py --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST 2>&1 | tee log.txt
python train.py --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST 2>&1 | tee log.txt
