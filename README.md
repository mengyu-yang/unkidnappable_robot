 # The Un-Kidnappable Robot: Acoustic Localization of Sneaking People

**Mengyu Yang, Patrick Grady, Samarth Brahmbhatt, Arun Balajee Vasudevan, Charles C. Kemp, James Hays** \
*Georgia Institute of Technology, Intel Labs, Carnegie Mellon University, Hello Robot Inc.*

[[arXiv](https://arxiv.org/abs/2310.03743)] [[Website](https://sites.google.com/view/unkidnappable-robot)]

## The Robot Kidnapper Dataset

We present the Robot Kidnapper Dataset, consisting of 4-channel audio paired with 360 degree RGB video frames. You can download the dataset here (both files required for training): [link1](https://www.dropbox.com/scl/fi/gc2jlq9n59gal6yu6hqwp/robot_kidnapper_dataset.tar?rlkey=npaj48uktw7lov7y3esw2uj09&dl=0), [link2](https://www.dropbox.com/scl/fi/p2g8ua6zu6rkqnp4bwxxl/robot_kidnapper_empty_dataset.tar?rlkey=mfvlk1iqyebtxruav1b5pyzmd&dl=0)

Please download and extract both tar files using the following commands: \
`tar -xvf robot_kidnapper_dataset.tar -C {directory to save folder in}` \
`tar -xvf robot_kidnapper_empty_dataset.tar -C {directory to save folder in}`

`robot_kidnapper_dataset.zip` contains a folder with the training and test examples and `robot_kidnapper_empty_dataset.zip` contains a folder with the empty room recordings used for data augmentation and background subtraction.

## Installation instructions

To run training, first create a new conda environment: \
`conda env create --name unkidnap_robot python=3.9`

Then, download the requirements: \
`pip install -r requirements.txt`

To train a model on the Robot Kidnapper dataset with default hyperparameters, first edit the dataset directory paths in `run_train.sh`. Under the `--data_dir` argument, add the path to the directory extracted from `robot_kidnapper_dataset.zip`. Under the `--empty_data_dir` argument, add the path to the directory extracted from `robot_kidnapper_empty_dataset.zip`. You can also change the test room used in the leave-one-out cross validation.

Then simply run `run_train.sh` to start training.

## Citation

If you'd like to cite our work, please use the following: 

```text
@article{
    yang2024unkidnappable,
    title={The Un-Kidnappable Robot: Acoustic Localization of Sneaking People},
    author={Yang, Mengyu and Grady, Patrick and Brahmbhatt, Samarth and Balajee Vasudevan, Arun and Kemp, Charles and Hays, James},
    journal={International Conference on Robotics and Automation (ICRA)},
    year={2024}
}
```
