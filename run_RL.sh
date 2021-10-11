path1=experiments_21/turn-level-DS/best_score_model
path2=experiments_21/all_turn-level-US-10-7_sd11_lr0.0001_bs4_ga8/best_score_model
python session.py -mode train\
 -cfg use_scheduler=True\
 lr=5e-5 batch_size=8 gradient_accumulation_steps=4\
 exp_no=rl-10-11-use_scheduler init_eval=False\
 training_batch_size=16\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=0 US_device=1