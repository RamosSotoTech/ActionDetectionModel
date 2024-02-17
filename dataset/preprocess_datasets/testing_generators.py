import os
from typing import Tuple, Generator, Callable, List, Any

import numpy as np
import cv2
from numpy import ndarray
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from typing_extensions import Optional
from utils.dir_utils import set_lists_dir, UCF_101_dir
from pathlib import Path
from random import shuffle
from dataset.preprocess_datasets.video_handling_utils import VideoProcessor, VideoBatchCreator, StopVideoIteration, ContinuousBatchVideoProcessor

with open(os.path.join(set_lists_dir, 'classInd.txt'), 'r') as f:
    class_labels = f.readlines()
class_labels = [label.strip() for label in class_labels]
class_label_dict = {int(label.split(' ')[0]) - 1: label.split(' ')[1] for label in class_labels}


def get_video_files_paths(fetch_path: str, file_name: str, data_path: str) -> List[Path]:
    with open(os.path.join(fetch_path, file_name), 'r') as f:
        video_files = f.readlines()
    video_files = [os.path.join(data_path, file) for file in video_files]
    video_files = [file.strip() for file in video_files]

    if 'train' in file_name:
        video_files = [file.split(' ')[0] for file in video_files]

    video_files = [Path(file) for file in video_files]
    return video_files


# get the label of the video file Path (file containing .avi files), return the label
def get_video_label(file_path: Path) -> int:
    parent_dir = file_path.parent
    label = parent_dir.name
    return label


def get_label_key(label: str) -> int:
    return list(class_label_dict.keys())[list(class_label_dict.values()).index(label)]


def get_label_from_key(key: str) -> int:
    return class_label_dict[key]


def get_video_path_label(file_path: Path) -> tuple[str, Any]:
    return str(file_path), to_categorical(get_label_key(get_video_label(file_path)), num_classes=len(class_label_dict))


def zip_video_files_labels(video_files: List[Path]) -> list[tuple[str, Any]]:
    return [get_video_path_label(file) for file in video_files]


def get_data_for_loading(indexes: List[int]) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    train_data = []
    test_data = []
    for index in indexes:
        train_video_files = get_video_files_paths(set_lists_dir, f'trainlist0{index}.txt', UCF_101_dir)
        test_video_files = get_video_files_paths(set_lists_dir, f'testlist0{index}.txt', UCF_101_dir)
        shuffle(train_video_files)
        shuffle(test_video_files)
        # concatenate
        train_data.append(train_video_files)
        test_data.append(test_video_files)
    train_video_files_labels = zip_video_files_labels(train_video_files)
    test_video_files_labels = zip_video_files_labels(test_video_files)

    return train_video_files_labels, test_video_files_labels


def video_dataset_generator(video_files_labels: List[Tuple[str, Any]], batch_size: int,
                            sequence_length: int, preprocessing: Optional[Callable] = None,
                            padding: str = 'post') -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    batch_videos = []  # To store batch sequences
    batch_labels = []  # To store labels for each sequence in the batch
    batch_indicators = []  # To store new video indicators for each sequence in the batch
    is_new_video = True  # Flag to indicate the start of a new video

    labels_iter = iter([label for _, label in video_files_labels])
    label = next(labels_iter)
    video_processor = VideoBatchCreator([video_file for video_file, _ in video_files_labels], preprocessing=preprocessing)
    sequence_frames = []

    while True:
        try:
            total_frames = []  # To store the total frames for each video in the batch
            for frame in video_processor:
                sequence_frames.append(frame)
                total_frames.append(len(sequence_frames))  # Add the total frames of the current video to the list

                if len(sequence_frames) == sequence_length:
                    # A full sequence is ready
                    max_frames = max(total_frames)  # Get the maximum number of frames among all videos in the batch
                    # Pre-pad each video sequence with zeros until they all have the same length as the maximum number of frames
                    sequence_frames = [np.pad(sequence, (max_frames - len(sequence), 0), 'constant') for sequence in sequence_frames]

                    batch_videos.append(np.array(sequence_frames))
                    batch_labels.append(label)

                    # Indicator: 1 at the start of a new video or sequence, 0 elsewhere
                    if is_new_video:
                        batch_indicators.append(np.array([1] + [0] * (sequence_length - 1)))
                        is_new_video = False  # Reset flag after the first sequence of the new video
                    else:
                        batch_indicators.append(
                            np.zeros(sequence_length))  # No new video indicator for subsequent sequences

                    sequence_frames = []  # Reset for the next sequence

                    if len(batch_videos) == batch_size:
                        # The batch is full, yield it
                        yield np.array(batch_videos), np.array(batch_labels), np.array(batch_indicators)
                        batch_videos, batch_labels, batch_indicators = [], [], []  # Reset for the next batch

                # Handle the last sequence if it's shorter than sequence_length
                if 0 < len(sequence_frames) < sequence_length:
                    # Pad the remaining frames as per the specified padding strategy
                    padding_frames = [np.zeros_like(sequence_frames[0]) for _ in range(sequence_length - len(sequence_frames))]
                    sequence_frames = sequence_frames + padding_frames if padding == 'post' else padding_frames + sequence_frames

                    batch_videos.append(np.array(sequence_frames))
                    batch_labels.append(label)
                    batch_indicators.append(
                        np.array([1] + [0] * (sequence_length - 1)) if is_new_video else np.zeros(sequence_length))
                    is_new_video = False  # Ensure flag is reset even if the last sequence triggers it

        except StopVideoIteration:
            # Handle the end of a video and the start of a new one
            is_new_video = True
            label = next(labels_iter, None)
            sequence_frames = []  # Reset for the next sequence
            continue
        except StopIteration:
            # Handle the end of all videos
            break

    # Handle the last batch if it's smaller than batch_size
    if len(batch_videos) > 0:
        # Pad the batch to reach batch_size
        while len(batch_videos) < batch_size:
            padding_sequence = np.zeros((sequence_length, *batch_videos[0].shape[1:]))
            batch_videos.append(padding_sequence)
            batch_labels.append(0)  # Use a dummy label for padding sequences
            batch_indicators.append(np.zeros(sequence_length))  # No new video indicator for padding sequences
        yield np.array(batch_videos), np.array(batch_labels), np.array(batch_indicators)

def get_dataset_generators(indexes: List[int], batch_size: int, preprocessing: Optional[Callable] = None,
                           sequence_length: int = 10, padding: str = 'post') -> tuple[
    Generator[tuple[ndarray, ndarray, ndarray], None, None], Generator[tuple[ndarray, ndarray, ndarray], None, None]]:
    train_data, test_data = get_data_for_loading(indexes)
    train_generator = video_dataset_generator(train_data, batch_size, sequence_length, preprocessing, padding)
    test_generator = video_dataset_generator(test_data, batch_size, sequence_length, preprocessing, padding)

    return train_generator, test_generator

def get_data_from_BatchVideoProcessor(video_files_labels: List[Tuple[str, Any]], batch_size: int, sequence_length: int,
                                     preprocessing: Optional[Callable] = None, padding: str = 'post') -> Tuple[
    Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None], Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]]:
    train_data, test_data = get_data_for_loading(video_files_labels)


    def preprocessing(frame: np.ndarray) -> np.ndarray:
        frame = preprocess_input(cv2.resize(frame, (320, 240)))
        frame = frame / 255.0
        return frame

    train_generator = ContinuousBatchVideoProcessor(train_data, batch_size, sequence_length, (240, 320, 3), preprocessing=preprocessing, padding=padding, training=True)
    test_generator = ContinuousBatchVideoProcessor(test_data, batch_size, sequence_length, (240, 320, 3), preprocessing=preprocessing, padding=padding, training=False)

    return train_generator, test_generator



if __name__ == "__main__":
    train_data, test_data = get_data_for_loading([1])
    batch_size = 2

    # Create the generator
    # train_generator = video_dataset_generator(train_data, batch_size, preprocessing=preprocess_input, padding='post', sequence_length=10)
    # test_generator = video_dataset_generator(test_data, batch_size, preprocessing=preprocess_input, padding='post', sequence_length=10)

    def preprocessing(frame: np.ndarray) -> np.ndarray:
        frame = preprocess_input(cv2.resize(frame, (320, 240)))
        frame = frame / 255.0
        return frame

    train_generator, test_generator = get_data_from_BatchVideoProcessor([1], batch_size, preprocessing=preprocess_input, sequence_length=10, padding='post')

    # Iterate through the generator
    for video_batch, labels, indicators in train_generator:
        # check if a new video was encountered in the batch
        print(indicators)
        print(video_batch.shape)
        print(labels.shape)
        # print the labels decoded using the argmax function
        print(np.argmax(labels, axis=1))
