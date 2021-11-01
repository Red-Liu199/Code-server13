path1=experiments_21/turn-level-DS/best_score_model
path2=experiments_21/turn-level-US/best_score_model
python session.py -mode train\
 -cfg use_scheduler=False\
 warmup_steps=2000\
 lr=2e-5\
 weight_decay_count=50\
 seed=1010\
 epoch_num=50\
 init_eval=False\
 training_batch_size=16\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=2 US_device=3\
 rl_for_bspn=True\
 non_neg_reward=False\
 clip_grad=False\
 ctrl_lr=True\
 rl_dial_per_epoch=128\
 joint_train_ds=True\
 joint_train_us=False\
 exp_no=rl-11-1-only-ds\
 add_rl_baseline=False\
 notes='lr 2e-5 rl_dial_per_epoch 128'