path=experiments_21/all_turn-level-US_sd11_lr0.0001_bs4_ga8/best_score_model
python pretrain.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    model_act=True dataset=1\
    eval_batch_size=24\
    train_us=True