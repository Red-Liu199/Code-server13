path1=/home/liuhong/myworkspace/experiments_21/all_turn-level-dst_sd11_lr0.0001_bs4_ga8/best_score_model
path2=/home/liuhong/myworkspace/experiments_21/all_turn-level-nlg_sd11_lr0.0001_bs4_ga8/best_score_model
python pretrain_test.py -mode test\
    -cfg gpt_path1=$path1 gpt_path2=$path2\
    cuda_device=$1\
    train_modular=True\
    modular=nlg\
    combine_eval=True\
    eval_batch_size=32