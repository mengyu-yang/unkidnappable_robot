#!/bin/bash

source ~/.bashrc
conda activate unkid_robot

data_dir='./robot_kidnapper_dataset'  # Change to your directory name
empty_data_dir='./robot_kidnapper_empty_dataset'  # Change to your directory name

# Choose from [howey-l4, howey-l5, molecular-1222, molecular-1201a, cherry-320, cherry-322, whitaker-1242, whitaker-1103]
# We evaluate performance using leave-one-out cross validation, where we train on all rooms except one room. This parameter specifies the room to leave out of training.
test_room='howey-l4'  

python train.py --log --data_dir $data_dir --empty_data_dir $empty_data_dir --log_spec --n_fft 512 --win_length 512 --hop_length 128 --num_epochs 75 --lr 1e-4 --batch_size 40 --weight_decay 1e-3 --acc_thresh 80 --background_sub --learn_backsub --backsub_w 0.8 --reg_w 1e-3 --reg_loss_fcn l1 --binary_depth --depth_threshold 1.7 --depth_posw 1.0 --depth_w 0.5 --cls_w 1.5 --subsample 2 --subsample_empty 1 --nostill --cls_posw 1.25 --empty_aug --empty_w 0.2 --empty_aug_prob 0.5 --spec_feat_nc 256 --mic_channels 0 1 2 3 --test_room $test_room
