path1=experiments_21/all_turn-level-ds-S_sd11_lr0.0001_bs8_ga4/best_score_model
path2=experiments_21/all_turn-level-us-111_sd11_lr0.0001_bs8_ga4/best_score_model
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=60\
 seed=11\
 epoch_num=60\
 init_eval=True\
 training_batch_size=8\
 rl_accumulation_steps=2\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=True\
 clip_grad=True\
 ctrl_lr=False\
 rl_dial_per_epoch=128\
 joint_train_ds=True\
 joint_train_us=True\
 add_rl_baseline=False\
 simple_reward=True\
 simple_training=False\
 exp_no=rl-11-16\
 transpose_batch=False\
 RL_ablation=False\
 reward_ablation=False\
 notes='DS pretrained on target context'