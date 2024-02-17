import os
from typing import Tuple, Generator, Callable, List, Any

import numpy as np
import cv2
from numpy import ndarray, dtype, generic
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from typing_extensions import Optional
import tensorflow as tf
from utils.dir_utils import set_lists_dir, UCF_101_dir
from pathlib import Path
from random import shuffle

with open(os.path.join(set_lists_dir, 'classInd.txt'), 'r') as f:
    class_labels = f.readlines()
class_labels = [label.strip() for label in class_labels]
class_label_dict = {int(label.split(' ')[0]) - 1: label.split(' ')[1] for label in class_labels}


# get path to the video files, return path to the video files
def get_video_files_paths(fetch_path: str, file_name: str, data_path: str) -> List[Path]:
    with open(os.path.join(fetch_path, file_name), 'r') as f:
        video_files = f.readlines()
    video_files = [os.path.join(data_path, file) for file in video_files]
    # remove the '\n' at the end of each line
    video_files = [file.strip() for file in video_files]

    if 'train' in file_name:
        video_files = [file.split(' ')[0] for file in video_files]

    # convert the list to a list of Path objects
    video_files = [Path(file) for file in video_files]
    return video_files


# get the label of the video file Path (file containing .avi files), return the label
def get_video_label(file_path: Path) -> int:
    # get the parent directory of the file
    parent_dir = file_path.parent
    # get the name of the parent directory
    label = parent_dir.name
    return label


# use class_label_dict to get key from value
def get_label_key(label: str) -> int:
    return list(class_label_dict.keys())[list(class_label_dict.values()).index(label)]


# get key from label using class_label_dict
def get_label_from_key(key: str) -> int:
    return class_label_dict[key]


# get tuple of video path and hot encoded label
def get_video_path_label(file_path: Path) -> tuple[str, Any]:
    return str(file_path), to_categorical(get_label_key(get_video_label(file_path)), num_classes=len(class_label_dict))


# zip the video files and their labels
def zip_video_files_labels(video_files: List[Path]) -> list[tuple[str, Any]]:
    return [get_video_path_label(file) for file in video_files]


# get the video frames from the video file
def get_video_sequences(video_path: Path, num_frames: int, frame_width: int, frame_height: int, window_size: int,
                        stride: int) -> Generator[Tuple[np.ndarray, Path,], None, None]:
    # Modified version of get_video_sequences to also yield the video_path with each sequence
    cap = cv2.VideoCapture(str(video_path))
    buffer = []  # Temporary buffer to hold frames for the current window

    while True:
        ret, frame = cap.read()
        if not ret:
            last_frame = buffer[-1]
            while len(buffer) < num_frames:
                buffer.append(last_frame)  # Pad with the last frame
            yield np.stack(buffer, axis=0)
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_width, frame_height))
        buffer.append(frame)

        # When the buffer size equals the window size, yield the sequence and update the buffer based on the stride
        if len(buffer) == window_size:
            yield np.stack(buffer, axis=0)  # Yields sequence and video_path
            buffer = buffer[stride:]

    cap.release()


# Update the generator function to use sequences
def make_sequence_generator(video_files_label: List[Tuple[Path, np.ndarray]], window_size: int, stride: int,
                            frame_width: int,
                            frame_height: int) -> Callable:
    def generator():
        for video_file, label in video_files_label:
            sequences_gen = get_video_sequences(video_file, window_size, frame_width, frame_height, window_size, stride)
            for sequence in sequences_gen:
                yield sequence, label  # Yields sequence, label

    return generator


def create_sequence_dataset(video_files_label: List[Tuple[Path, np.ndarray]], window_size: int, stride: int,
                            frame_width: int,
                            frame_height: int) -> tf.data.Dataset:
    generator = make_sequence_generator(video_files_label, window_size, stride, frame_width, frame_height)
    # return tensor with frames and prediction (hot encoded label)
    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(window_size, frame_width, frame_height, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(101,), dtype=tf.int32),
    ))


# get generator for provided int in dataset to load combined
def get_dataset_generators(dataset: List[int], batch_size: int,
                           preprocessing: Optional[Callable] = None,
                           sequence_length: Optional[int] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def load_data(index: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_video_files = get_video_files_paths(set_lists_dir, f'trainlist0{index}.txt', UCF_101_dir)
        test_video_files = get_video_files_paths(set_lists_dir, f'testlist0{index}.txt', UCF_101_dir)
        shuffle(train_video_files)
        shuffle(test_video_files)
        train_video_files_labels = zip_video_files_labels(train_video_files)
        test_video_files_labels = zip_video_files_labels(test_video_files)
        train_dataset = create_sequence_dataset(train_video_files_labels, sequence_length, 8, 224, 224)
        test_dataset = create_sequence_dataset(test_video_files_labels, sequence_length, 8, 224, 224)

        if preprocessing:
            train_dataset = train_dataset.map(lambda x, y: (preprocessing(x), y))
            test_dataset = test_dataset.map(lambda x, y: (preprocessing(x), y))

        if sequence_length:
            def sequence_padding(x, y):
                shape = tf.shape(x)[0]  # Get the current sequence length
                sequence_length_tensor = tf.cast(sequence_length, dtype=tf.int32)  # Desired sequence length as a tensor

                # Calculate padding size, ensuring it's not negative
                sequence_length_minus_shape = tf.maximum(sequence_length_tensor - shape, 0)

                # Create paddings tensor, ensuring no negative values
                paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 0]])  # Assuming 'x' has shape [seq_len, H, W, C]
                paddings = tf.tensor_scatter_nd_update(paddings, [[0, 1]], [
                    sequence_length_minus_shape])  # Update padding for the sequence length dimension

                # Apply padding
                x = tf.pad(x, paddings)
                return x, y

            train_dataset = train_dataset.map(sequence_padding)
            test_dataset = test_dataset.map(sequence_padding)

        return train_dataset, test_dataset

    train_datasets, test_datasets = zip(*(load_data(index) for index in dataset))

    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == "__main__":
    # get the video files for the train and test datasets
    train_video_files = get_video_files_paths(set_lists_dir, 'trainlist01.txt', UCF_101_dir)
    test_video_files = get_video_files_paths(set_lists_dir, 'testlist01.txt', UCF_101_dir)

    # zip the video files and their labels
    train_video_files_labels = zip_video_files_labels(train_video_files)
    test_video_files_labels = zip_video_files_labels(test_video_files)

    # create the datasets
    # train_dataset = create_sequence_dataset(train_video_files_labels, 16, 8, 224, 224)
    # test_dataset = create_sequence_dataset(test_video_files_labels, 16, 8, 224, 224)

    train_dataset, test_dataset = get_dataset_generators([1], batch_size=4, preprocessing=preprocess_input, sequence_length=60)
    # take randomly 10 items from train_dataset
    for sequence, label in train_dataset.take(10):
        print("Sequence shape:", sequence.shape)
        print("Label:", label.numpy())

    # print(train_video_files_labels[:5])
    # print(test_video_files_labels[:5])
    # print(class_label_dict)
