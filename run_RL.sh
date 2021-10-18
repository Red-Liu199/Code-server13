path1=experiments_21/turn-level-DS/best_score_model
path2=experiments_21/all_turn-level-US-10-7_sd11_lr0.0001_bs4_ga8/best_score_model
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=1e-6 batch_size=8 gradient_accumulation_steps=4\
 exp_no=rl-10-15-neg init_eval=True\
 training_batch_size=16\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=2 US_device=3\
 rl_for_bspn=True\
 non_neg_reward=False