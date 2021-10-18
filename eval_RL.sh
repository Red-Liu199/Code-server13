#path1=experiments_21/turn-level-DS/best_score_model
#path2=experiments_21/all_turn-level-US-10-7_sd11_lr0.0001_bs4_ga8/best_score_model
path1=/home/liuhong/myworkspace/RL_exp/rl-10-15-neg/best_DS
path2=/home/liuhong/myworkspace/RL_exp/rl-10-15-neg/best_US
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1