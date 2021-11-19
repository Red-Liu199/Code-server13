path1=/home/liuhong/myworkspace/experiments_21/all_DST_sd11_lr0.0001_bs8_ga4/best_score_model
path2=/home/liuhong/myworkspace/experiments_21/all_DM_sd11_lr0.0001_bs8_ga4/best_score_model
path3=/home/liuhong/myworkspace/experiments_21/all_NLG_sd11_lr0.0001_bs8_ga4/best_score_model
python pretrain.py -mode test\
    -cfg gpt_path1=$path1 gpt_path2=$path2 gpt_path3=$path3\
    cuda_device=$1\
    train_modular=True\
    modular=nlg\
    combine_eval=True\
    eval_batch_size=32