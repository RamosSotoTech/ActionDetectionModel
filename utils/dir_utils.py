import os

utils_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(utils_dir, os.pardir)
dataset_dir = os.path.join(root_dir, "dataset")
model_dir = os.path.join(root_dir, "models")
preprocessed_datasets_dir = os.path.join(dataset_dir, "preprocess_datasets")
UCF_101_dir = os.path.join(dataset_dir, "UCF-101")
set_lists_dir = os.path.join(dataset_dir, "ucfTrainTestlist")
