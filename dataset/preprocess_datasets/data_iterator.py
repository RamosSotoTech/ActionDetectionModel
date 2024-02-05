import os
from typing import Tuple, Generator, Callable, List

import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from typing_extensions import Optional

from utils.dir_utils import preprocessed_datasets_dir, UCF_101_dir

np.random.seed(13245)


class VideoException(Exception):
    pass


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(frame, target_size)


def load_video(video_path: str, sequence_length: int = 10, preprocessing: Callable = resize_frame,
               step: int = 1) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if preprocessing:
            frame = preprocessing(frame)
        frames.append(frame)
        if len(frames) == sequence_length:
            yield np.array(frames)
            frames = frames[step:]
    cap.release()


# todo: create a generator that can be used by k-fold cross validation
def data_generator(dataset_paths: List[str], batch_size: int = 4, preprocessing: Callable = resize_frame,
                   sequence_length: int = 10, step: int = 1, include_labels: bool = True, num_classes: int = 101) -> \
        Tuple[np.ndarray, np.ndarray]:
    while True:
        batch_sequences = []
        batch_labels = []
        for path in dataset_paths:
            data = np.load(path, allow_pickle=True)
            np.random.shuffle(data)
            for item in data:
                video_path, label_idx = item[0], item[2]
                video_full_path = os.path.join(UCF_101_dir, video_path)
                for sequence in load_video(video_full_path, sequence_length=sequence_length,
                                           preprocessing=preprocessing, step=step):
                    batch_sequences.append(sequence)
                    if include_labels:
                        label = to_categorical(label_idx - 1, num_classes=num_classes)
                        batch_labels.append(label)
                    if len(batch_sequences) == batch_size:
                        yield (np.array(batch_sequences), np.array(batch_labels)) if include_labels else np.array(
                            batch_sequences)
                        batch_sequences, batch_labels = [], []

        if batch_sequences:
            yield (np.array(batch_sequences), np.array(batch_labels)) if include_labels else np.array(batch_sequences)


def get_dataset_generators(dataset_numbers: List[int], batch_size: int = 4,
                           preprocessing: Optional[Callable] = resize_frame,
                           sequence_length: int = 10, step: int = 1, num_classes: int = 101) -> Tuple[
    Generator, Generator]:
    train_paths = [os.path.join(preprocessed_datasets_dir, f'train_dataset_{i}.npy') for i in dataset_numbers]
    test_paths = [os.path.join(preprocessed_datasets_dir, f'test_dataset_{i}.npy') for i in dataset_numbers]
    train_gen = data_generator(train_paths, batch_size, preprocessing, sequence_length, step, True, num_classes)
    test_gen = data_generator(test_paths, batch_size, preprocessing, sequence_length, step, True, num_classes)
    return train_gen, test_gen


def get_train_test_size(dataset_numbers: List[int]) -> Tuple[int, int]:
    train_samples = 0
    test_samples = 0
    for num in dataset_numbers:
        train_path = os.path.join(preprocessed_datasets_dir, f'train_dataset_{num}.npy')
        test_path = os.path.join(preprocessed_datasets_dir, f'test_dataset_{num}.npy')

        if os.path.isfile(train_path):
            data = np.load(train_path, allow_pickle=True)
            train_samples += len(data)

        if os.path.isfile(test_path):
            data = np.load(test_path, allow_pickle=True)
            test_samples += len(data)

    return train_samples, test_samples


if __name__ == "__main__":
    dataset_numbers = [1, 2, 3]
    train_generator, test_generator = get_dataset_generators(dataset_numbers, batch_size=4, preprocessing=resize_frame)

    for batch_videos, batch_labels in train_generator:
        print(f"Batch shape: {batch_videos.shape}, Labels: {batch_labels}")

    for batch_videos, batch_labels in test_generator:
        print(f"Batch shape: {batch_videos.shape}, Labels: {batch_labels}")
        break

    print("For loop complete.")
