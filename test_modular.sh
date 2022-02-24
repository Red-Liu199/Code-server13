path1=experiments_21/all_DST-1-5_sd11_lr0.0001_bs8_ga4/best_score_model
#path2=experiments_21/all_DM-1-5_sd11_lr0.0001_bs8_ga4/best_score_model
path2=RL_exp/RL-1-6-fix_DST/best_DS
python pretrain.py -mode test\
    -cfg gpt_path1=$path1 gpt_path2=$path2\
    cuda_device=$1\
    train_modular=True\
    modular=dst\
    combine_eval=True\
    eval_batch_size=32