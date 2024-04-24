 # The Un-Kidnappable Robot: Acoustic Localization of Sneaking People

[[arXiv](https://arxiv.org/abs/2310.03743)] [[Website](https://sites.google.com/view/unkidnappable-robot)]

## The Robot Kidnapper Dataset

We collect the Robot Kidnapper Dataset, consisting of 4-channel audio paired with 360 degree RGB video frames. You can download the dataset here: [link]()

Please download and extract both zip files. `robot_kidnapper_dataset.zip` contains a folder with the training and test examples and `robot_kidnapper_empty_dataset.zip` contains a folder with the empty room recordings used for data augmentation and background subtraction.

## Installation instructions

To run training, first create a new conda environment: \
`conda env create --name {you-env-name} python=3.9`

Then, download the requirements: \
`pip install -r requirements.txt`

To train a model on the Robot Kidnapper dataset with default hyperparameters, first edit the dataset directory paths in `run_train.sh`. Under the `--data_dir` argument, add the path to the directory extracted from `robot_kidnapper_dataset.zip`. Under the `--empty_data_dir` argument, add the path to the directory extracted from `robot_kidnapper_empty_dataset.zip`. 

Then run `run_train.sh` by either copying the contents into the terminal or running the file directly.

