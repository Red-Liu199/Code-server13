path1=/home/liuhong/myworkspace/RL_exp/rl-11-1-revise-reward/best_DS
path2=/home/liuhong/myworkspace/experiments_21/turn-level-US/best_score_model
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1

path1=/home/liuhong/myworkspace/experiments_21/turn-level-DS/best_score_model
path2=/home/liuhong/myworkspace/RL_exp/rl-11-1-revise-reward/best_US
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1