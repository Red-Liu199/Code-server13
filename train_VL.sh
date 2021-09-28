# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script runs the Semi-VL (variational learning) experiment on MultiWOZ.
# Before Semi-VL, make sure that you have pretrained your model 
# (both generative and inference model) on supervised data.
# The two models in paths will be loaded as initialization model.
# The proportion you choose must be consistent with that during pretraining stage.

ratio=10

path1=experiments_21/all_pre_${ratio}_act_sd11_lr0.0001_bs2_ga16/best_loss_model

path2=experiments_21/all_pre_pos${ratio}_act_sd11_lr0.0001_bs2_ga16/best_loss_model

python train_semi.py -mode semi_VL\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=40\
    cuda_device=2,3\
    spv_proportion=$ratio\
    model_act=True\
    dataset=1\
    exp_no=VL-test
