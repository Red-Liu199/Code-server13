path1=experiments_21/DS_base/best_score_model
path2=experiments_21/US_base/best_loss_model
python session4RL.py -cfg mode=test cuda_device=1,1 DS_path=$path1 US_path=$path2