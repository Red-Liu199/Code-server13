path=experiments_21/US-with-NLU/best_score_model
python main_turn_level_us.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    model_act=True dataset=1\
    eval_batch_size=32\
    train_us=True\
    user_nlu=True\
    strict_eval=True