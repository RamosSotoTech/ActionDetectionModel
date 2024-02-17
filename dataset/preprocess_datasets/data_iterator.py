import os
from typing import Tuple, Generator, Callable, List

import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from typing_extensions import Optional
import tensorflow as tf
from utils.dir_utils import preprocessed_datasets_dir, UCF_101_dir

np.random.seed(13245)


class VideoException(Exception):
    pass


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    return frame / 255.0


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(frame, target_size)


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    frame = cv2.resize(frame, target_size)
    # convert the frame to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = normalize_frame(frame)
    frame = preprocess_input(frame)
    return frame


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
def data_generator(dataset_paths: List[str], batch_size: int = 4, preprocessing: Callable = preprocess_frame,
                   sequence_length: int = 10, step: int = 1, include_labels: bool = True, num_classes: int = 101, data_augmentation=False,
                   max_videos: Optional[int] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    video_count = 0
    while True:
        batch_sequences = []
        batch_labels = []
        for path in dataset_paths:
            data = np.load(path, allow_pickle=True)
            np.random.shuffle(data)
            for item in data:
                if max_videos is not None and video_count >= max_videos:
                    return
                video_count += 1
                video_path, label_idx = item[0], item[2]
                video_full_path = os.path.join(UCF_101_dir, video_path)
                # preselected data augmentation for entire video sequence
                if data_augmentation:
                    flip = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) == 1
                    delta = tf.random.uniform(shape=[], minval=-0.1, maxval=0.1)
                    contrast_factor = tf.random.uniform(shape=[], minval=0.1, maxval=0.2)
                    rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
                for sequence in load_video(video_full_path, sequence_length=sequence_length,
                                           preprocessing=preprocessing, step=step):
                    if data_augmentation:
                        if flip:
                            sequence = tf.image.flip_left_right(sequence)
                        sequence = tf.image.adjust_brightness(sequence, delta)
                        sequence = tf.image.adjust_contrast(sequence, contrast_factor)
                        sequence = tf.image.rot90(sequence, k=rotation)
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
                           preprocessing: Optional[Callable] = preprocess_frame,
                           sequence_length: int = 10, step: int = 1, num_classes: int = 101) -> Tuple[
    Generator, Generator]:
    train_paths = [os.path.join(preprocessed_datasets_dir, f'train_dataset_{i}.npy') for i in dataset_numbers]
    test_paths = [os.path.join(preprocessed_datasets_dir, f'test_dataset_{i}.npy') for i in dataset_numbers]
    # apply data augmentation to the training set
    train_gen = data_generator(train_paths, batch_size, preprocessing, sequence_length, step, True, num_classes, data_augmentation=True)
    test_gen = data_generator(test_paths, batch_size, preprocessing, sequence_length, step, True, num_classes, data_augmentation=False)
    return train_gen, test_gen


# def get_tensorflow_dataset(dataset_numbers: List[int], batch_size: int = 4,
#                             preprocessing: Optional[Callable] = preprocess_frame,
#                             sequence_length: int = 10, step: int = 1, num_classes: int = 101) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
#     dataset = tf.data.Dataset.from_generator(
#         lambda: load_video(video_path, sequence_length, preprocess_frame, step),
#         output_types=tf.float32,  # Ensure this matches your actual frame data type
#         output_shapes=(sequence_length, 224, 224, 3)  # Adjust based on your frame size and number of channels
#     )
#
#     # Apply batching, prefetching, and possibly interleave if working with multiple videos
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset



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


# if __name__ == "__main__":
#     dataset_numbers = [1, 2, 3]
#     subset_fraction = 0.2 # Update here: can be adjusted according to needs
#     # Update here: pass the smaller_fraction parameter to get_dataset_generators
#     train_generator, test_generator = get_dataset_generators(dataset_numbers, batch_size=4, subset_fraction=subset_fraction, preprocessing=preprocess_frame)
#
#     for batch_videos, batch_labels in train_generator:
#         print(f"Batch shape: {batch_videos.shape}, Labels: {batch_labels}")
#
#     for batch_videos, batch_labels in test_generator:
#         print(f"Batch shape: {batch_videos.shape}, Labels: {batch_labels}")
#         break
#
#     print("For loop complete.")

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import gc


from PIL import Image, ImageTk
import numpy as np
import cv2
import gc

class GUI_Video_from_get_data_generators:
    def __init__(self, iterator):
        self.iterator = iter(iterator)  # Ensure it's an iterator
        self.root = tk.Tk()
        self.root.geometry('800x600')  # Adjusted for better visibility

        self.label = tk.Label(self.root)
        self.label.pack()

        self.root.bind('<space>', lambda event: self.video_loop(skip_frame=True))  # Skip one frame
        self.root.bind('<Control-Key>', lambda event: self.next_video())  # Skip to next batch

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.next_batch()  # Initialize the first batch

        self.root.mainloop()

    def next_batch(self):
        try:
            batch, _ = next(self.iterator)  # Assuming the iterator yields (batch_data, batch_labels)
            self.current_batch = iter(batch)  # Make an iterator from the batch
            self.video_loop()
        except StopIteration:
            print("End of dataset")
            self.root.destroy()

    def video_loop(self, skip_frame=False):
        try:
            if skip_frame:
                # Attempt to skip a frame; if no more frames, go to next batch
                try:
                    next(self.current_batch)
                except StopIteration:
                    self.next_batch()
                    return

            batched_frame = next(self.current_batch)
            # Check if the frame is batched and extract the single frame
            if isinstance(batched_frame, np.ndarray) and batched_frame.shape[0] == 1:
                frame = batched_frame[0]  # Extract the single frame from the batch

                if len(frame.shape) == 3:  # Ensure the extracted frame has 3 dimensions
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = ImageTk.PhotoImage(frame)
                    self.label.configure(image=frame)
                    self.label.image = frame  # Keep the reference
                    self.root.after(100, self.video_loop)  # Schedule next frame update
                else:
                    print("Extracted object is not a proper image frame.")
            else:
                print("Encountered an object that is not a batched single image frame.")

        except StopIteration:
            print("Reached the end of the current batch, moving to the next one.")
            self.next_batch()

    def next_video(self):
        self.next_batch()  # Jump to the next batch

    def on_closing(self):
        self.root.destroy()



def data_iterator():
    # Create a generator for the UCF-101 dataset
    train_generator = get_dataset_generators([1], batch_size=4, preprocessing=None, sequence_length=1)[0]
    return train_generator


if __name__ == "__main__":
    GUI_Video_from_get_data_generators(data_iterator())