from typing import List, Any

import cv2
import numpy as np
import tensorflow as tf


class FrameFetchingStrategy:
    def fetch_frame(self, video):
        raise NotImplementedError("Subclass must implement the fetch_frame method.")


class FetchEveryFrameStrategy(FrameFetchingStrategy):
    def fetch_frame(self, video):
        ret, frame = video.read()
        return ret, frame


class FetchEveryNthFrameStrategy(FrameFetchingStrategy):
    def __init__(self, n=1):
        self.n = n  # Skip every n frames
        self.counter = 0

    def fetch_frame(self, video):
        ret, frame = None, None
        while self.counter < self.n:
            ret, frame = video.read()
            self.counter += 1
        self.counter = 0  # Reset counter after nth frame is fetched
        return ret, frame


class VideoProcessor:
    def __init__(self, file_path, preprocessing=None, target_shape=None, training=False):
        self.video = None
        self.file_path = file_path
        self.preprocessing = preprocessing
        self.target_shape = target_shape
        self.training = training
        if self.training:
            self.randomize_augment_parameters()
        try:
            self.video = cv2.VideoCapture(self.file_path)
            if not self.video.isOpened():
                raise ValueError("Could not open video file.")
            # Extract metadata
            self.original_frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        except cv2.error as e:
            raise FileNotFoundError(f"{e}: Could not open video file: {self.file_path}")
        finally:
            self.current_frame = 0

    def get_metadata(self):
        return {
            'original_frame_rate': self.original_frame_rate,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'total_frames': self.total_frames,
        }

    def fetch_sequence(self, num_frames):
        sequence = []
        while len(sequence) < num_frames and self.current_frame < self.total_frames:
            ret, frame = self.video.read()
            if not ret:
                break  # Stop if no frame is read
            if self.preprocessing is not None:
                frame = self.preprocessing(frame)
            if self.training:
                frame = self.apply_augmentation(frame)
            sequence.append(frame)
            self.current_frame += 1

        return tf.stack(sequence)

    def standardize_fetch(self, fps_target, num_frames):
        sequence = []
        frame_interval = self.original_frame_rate / fps_target  # Calculate the interval between frames to fetch

        target_frame_indices = [int(round(self.current_frame + i * frame_interval)) for i in range(num_frames)]
        last_fetched_index = -1  # Keep track of the last fetched frame index

        for frame_index in target_frame_indices:
            # Ensure we do not fetch the same frame twice or go backward due to rounding, and do not exceed total frames
            if last_fetched_index < frame_index < self.total_frames:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.video.read()
                if not ret:
                    break  # Stop if no frame is read
                if self.preprocessing is not None:
                    frame = self.preprocessing(frame)
                if self.training:
                    frame = self.apply_augmentation(frame)
                sequence.append(frame)
                last_fetched_index = frame_index
                self.current_frame = frame_index + 1  # Update the current frame to the last one fetched

        return sequence

    def standardize_fetch_generator(self, fps_target, num_frames):
        frame_interval = self.original_frame_rate / fps_target  # Calculate the interval between frames to fetch

        while True:  # The generator will continue until it can't yield a full sequence
            sequence = []
            start_frame_index = self.current_frame

            for i in range(num_frames):
                target_frame_index = int(round(start_frame_index + i * frame_interval))
                if target_frame_index >= self.total_frames:
                    return  # Stop the generator if there aren't enough frames left

                self.video.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                ret, frame = self.video.read()
                if not ret:
                    return  # Stop the generator if unable to read the frame

                if self.preprocessing is not None:
                    frame = self.preprocessing(frame)
                if self.training:
                    frame = self.apply_augmentation(frame)
                sequence.append(frame)
            self.current_frame = int(round(start_frame_index + num_frames * frame_interval))  # Update current_frame

            # Stack the sequence and remove the first singleton dimension (if present)
            sequence_tensor = tf.stack(sequence)
            if sequence_tensor.shape[0] == 1:  # Check if the first dimension is singleton
                sequence_tensor = tf.squeeze(sequence_tensor, axis=0)

            yield sequence_tensor

    def randomize_augment_parameters(self):
        self.flip = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32).numpy() == 1
        self.delta = tf.random.uniform(shape=[], minval=-0.05,
                                       maxval=0.05).numpy()  # More conservative range for brightness
        self.contrast_factor = tf.random.uniform(shape=[], minval=0.9,
                                                 maxval=1.1).numpy()  # Narrower range for contrast
        self.to_grayscale = tf.random.uniform(shape=[], minval=0, maxval=2,
                                              dtype=tf.int32).numpy() == 1  # Randomize grayscale conversion

    def apply_augmentation(self, frame_tensor):
        # Normalize the frame tensor
        frame_tensor = tf.cast(frame_tensor, tf.float32) / 255.0

        # Conditionally flip the image horizontally
        frame_tensor = tf.cond(self.flip > 0,
                               lambda: tf.image.flip_left_right(frame_tensor),
                               lambda: frame_tensor)

        # Adjust brightness and contrast
        frame_tensor = tf.image.adjust_brightness(frame_tensor, self.delta)
        frame_tensor = tf.image.adjust_contrast(frame_tensor, self.contrast_factor)

        # Conditionally convert to grayscale and then back to RGB
        def to_grayscale_and_back(x):
            grayscale = tf.image.rgb_to_grayscale(x)
            return tf.image.grayscale_to_rgb(grayscale)

        frame_tensor = tf.cond(self.to_grayscale > 0,
                               lambda: to_grayscale_and_back(frame_tensor),
                               lambda: frame_tensor)

        # Clip values to ensure they are within [0, 1]
        frame_tensor = tf.clip_by_value(frame_tensor, 0.0, 1.0)

        return frame_tensor

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()

    def number_remaining_frames(self):
        return self.total_frames - self.current_frame


class StopVideoIteration(Exception):
    pass


class ContinuousVideoProcessor:

    def __init__(self, file_paths_labels, frames_per_batch, shape, preprocessing=None, padding_frame=None,
                 training=True):
        self.preprocessing = preprocessing
        self.frames_per_batch = frames_per_batch
        self.frame_height, self.frame_width, self.color_channels = shape
        if padding_frame is None:
            self.padding_frame = np.zeros((self.frame_height, self.frame_width, self.color_channels), dtype=np.uint8)

        self.video_paths_iter = iter(file_paths_labels)
        self.paths_completed = len(file_paths_labels) == 0
        self.finished_current_video = False
        self.training = training

        def videoProcessorGenerator():
            while True:
                try:
                    video_path, labels = next(self.video_paths_iter)
                    yield VideoProcessor(video_path, preprocessing=self.preprocessing, training=self.training), labels
                except StopIteration:
                    self.paths_completed = True
                    yield None, None
                except FileNotFoundError as e:
                    print(e)
                    continue

        self.video_generator = videoProcessorGenerator()
        self.current_video_processor, self.current_labels = next(self.video_generator)

    def __iter__(self):
        return self

    def __next__(self):
        while True:  # Keep trying until a valid sequence is found or all videos are processed.
            if self.paths_completed:
                raise StopIteration("All videos processed.")

            if self.finished_current_video or self.current_video_processor.number_remaining_frames() < self.frames_per_batch:
                # Move to the next video if the current one is finished or doesn't have enough frames
                try:
                    self.current_video_processor, self.current_labels = next(self.video_generator)
                    self.finished_current_video = False
                except StopIteration:
                    self.paths_completed = True
                    raise StopIteration("All videos processed.")
                continue  # Check the next video processor

            # Fetch the entire sequence of frames
            sequence_frames = self.current_video_processor.fetch_sequence(self.frames_per_batch)

            if len(sequence_frames) == self.frames_per_batch:
                sequence_frames_np = tf.stack(sequence_frames)  # Convert list of sequences to tensor
                sequence_frames_np = tf.reshape(sequence_frames_np, [-1, self.frame_height, self.frame_width,
                                                                     self.color_channels])
                return sequence_frames_np, tf.convert_to_tensor(
                    self.current_labels)  # Return tensor instead of np.array
            else:
                # If fetched frames are less than frames_per_batch, mark the current video as finished and loop again
                self.finished_current_video = True


def display_sequence(sequence):
    frame_height, frame_width = 240, 320  # Desired display resolution

    for frame in sequence:
        resized_frame = cv2.resize(frame.numpy(),
                                   (frame_width, frame_height))  # Resize frame for consistent display size
        cv2.imshow('Video Sequence', resized_frame)  # Display the frame
        key = cv2.waitKey(100)  # Wait for 100 ms between frames
        if key == 27:  # Exit loop if ESC is pressed
            break


if __name__ == "__main__":
    from dataset.preprocess_datasets.testing_generators import get_data_for_loading

    video_files = get_data_for_loading([1])[0]
    frames_per_batch = 10
    frame_shape = (240, 320, 3)  # Height, width, channels
    video_processor = ContinuousVideoProcessor(video_files, frames_per_batch, frame_shape)
    for sequence, labels in video_processor:
        print(sequence.shape)
        print(np.argmax(labels, axis=0))
        display_sequence(sequence)
    cv2.destroyAllWindows()

    print("All videos processed.")
