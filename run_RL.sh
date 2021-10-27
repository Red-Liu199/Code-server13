path1=experiments_21/turn-level-DS/best_score_model
path2=experiments_21/all_turn-level-US-10-7_sd11_lr0.0001_bs4_ga8/best_score_model
python session.py -mode train\
 -cfg use_scheduler=False\
 warmup_steps=2000\
 lr=2e-5\
 epoch_num=40\
 exp_no=rl-10-26\
 init_eval=False\
 training_batch_size=16\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=2 US_device=3\
 rl_for_bspn=True\
 non_neg_reward=False\
 clip_grad=False\
 ctrl_lr=True\
 notes='No scheduler with lr=2e-5 and weight_decay_count=4, to prove the scheduler is the cause of deterioration'