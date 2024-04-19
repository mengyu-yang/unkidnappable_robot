# The Un-Kidnappable Robot: Acoustic Localization of Sneaking People

[[arXiv](https://arxiv.org/abs/2310.03743)] [[Website](https://sites.google.com/view/unkidnappable-robot)]

## The Robot Kidnapper Dataset

We collect ...

## Installation instructions

To run training, first create a new conda environment: \
`conda env create --name {you-env-name} python=3.9`

Then, download the requirements: \
`pip install -r requirements.txt`



To run test experiment: 

1. Request an interactive gpu with `salloc --gres=gpu:1 --cpus-per-task=7 -p overcap -J "inter" --exclude=bmo,hal,t1000 --qos debug`

2. Download miniconda and create a conda environment using the `environment.yml` file

3. Activate the conda environment and run the training script: `python train.py --proj_name unkidnappable_robot --exp_name test --log_spec --n_fft 512 --win_length 512 --hop_length 128 --num_epochs 75 --lr 1e-4 --batch_size 8 --weight_decay 1e-3 --acc_thresh 80 --background_sub --learn_backsub --backsub_w 0.8 --reg_w 1e-3 --reg_loss_fcn l1 --binary_depth --depth_threshold 1.7 --depth_posw 1.0 --depth_w 0.5 --cls_w 1.5 --subsample 2 --subsample_empty 1 --nostill --cls_posw 1.25 --empty_aug --empty_w 0.2 --empty_aug_prob 0.5 --spec_feat_nc 256 --mic_channels 0 1 2 3 --test_room cherry-320`
