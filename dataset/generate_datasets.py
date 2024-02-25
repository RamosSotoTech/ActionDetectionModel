import os
from typing import Dict, List, NoReturn, Tuple

import numpy as np
import pandas as pd
from tensorflow.data import Dataset

from utils.dir_utils import dataset_dir, preprocessed_datasets_dir, set_lists_dir


def read_classInd() -> Dict[str, int]:
    classInd = {}
    with open(os.path.join(set_lists_dir, 'classInd.txt')) as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) != 2:
                print(f"Unexpected format: {line}")
            else:
                number, label = split_line
                # convert label to lowercase
                label = label.lower()
                classInd[label] = int(number)
    return classInd


def _extract_data(parts: list, include_class_number: bool) -> dict:
    # Function to extract video_path, class_label, and class_number from parts
    relative_video_path = parts[0]
    class_label = relative_video_path.split('/')[1].split('_')[1].lower()
    data = {'video_path': relative_video_path, 'class_label': class_label}
    if include_class_number:
        class_number = parts[1]
        data['class_number'] = int(class_number)
    else:
        classInd = read_classInd()
        data['class_number'] = classInd[class_label]

    return data


def read_ucf_list(file_path: str, include_class_number: bool = True) -> pd.DataFrame:
    # Reads a UCF list file and returns a dataframe with video paths and optionally class labels
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if include_class_number and len(parts) != 2:
                print(f"Skipping line in {file_path}: '{line.strip()}' - expected video path and class number.")
                continue
            data.append(_extract_data(parts, include_class_number))
    return pd.DataFrame(data)


def generate_datasets(list_base_path: str, train_list_files: List[str], test_list_files: List[str]) -> Tuple[
    List[Dataset], List[Dataset]]:
    # Generates train and test datasets from the UCF list files,
    train_datasets = [read_ucf_list(os.path.join(list_base_path, f), include_class_number=True) for f in
                      train_list_files]
    test_datasets = [read_ucf_list(os.path.join(list_base_path, f), include_class_number=False) for f in
                     test_list_files]
    return train_datasets, test_datasets


def save_datasets(datasets: List[pd.DataFrame], base_path: str, prefix: str) -> NoReturn:
    for i, dataset in enumerate(datasets):
        # Convert DataFrame to NumPy array
        array = dataset.to_numpy()
        # Construct file path
        file_path = os.path.join(base_path, f"{prefix}_dataset_{i + 1}.npy")
        # Save array to .npy file
        np.save(file_path, array, allow_pickle=True)
        print(f"Saved {file_path}")


if __name__ == "__main__":
    list_base_path = os.path.join(dataset_dir, 'ucfTrainTestlist')
    train_list_files = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']
    test_list_files = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt']
    train_datasets, test_datasets = generate_datasets(list_base_path, train_list_files, test_list_files)

    save_datasets(train_datasets, preprocessed_datasets_dir, 'train')
    save_datasets(test_datasets, preprocessed_datasets_dir, 'test')
    print("Datasets saved.")
