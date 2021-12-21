ratio=10
path1=/home/liuhong/myworkspace/experiments_21/all_turn_level_10_sd11_lr0.0001_bs8_ga4/best_score_model
path2=/home/liuhong/myworkspace/experiments_21/all_turn_level_10_pos_sd11_lr0.0001_bs8_ga4/best_loss_model

python main_turn_level.py -mode semi_VL\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    cuda_device=2,3\
    spv_proportion=$ratio\
    exp_no=VL-turn-level-${ratio}\
    turn_level=True\
    debugging=True
