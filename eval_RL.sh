path1=/home/liuhong/myworkspace/experiments_21/turn-level-DS/best_score_model
path2=/home/liuhong/myworkspace/experiments_21/all_turn-level-us-111_sd11_lr0.0001_bs8_ga4/best_score_model
#path1=/home/liuhong/myworkspace/RL_exp/rl-ablation-no-scheduler/best_DS
#path2=/home/liuhong/myworkspace/RL_exp/rl-ablation-no-scheduler/best_US
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=True\
 beam_search=True beam_size=8\
 DS_device=$1 US_device=$1