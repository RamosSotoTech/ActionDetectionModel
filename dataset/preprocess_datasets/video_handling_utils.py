from typing import List, Any

import cv2
import numpy as np
import tensorflow as tf


class VideoProcessor:
    def __init__(self, file_path, preprocessing=None):
        self.video = None
        self.file_path = file_path
        self.preprocessing = preprocessing
        try:
            self.video = cv2.VideoCapture(self.file_path)
            if not self.video.isOpened():
                raise ValueError("Could not open video file.")
            # Extract metadata
            self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.color_channels = int(self.video.get(cv2.CAP_PROP_CHANNEL))
        except cv2.error as e:
            raise FileNotFoundError(f"{e}: Could not open video file: {self.file_path}")
        finally:
            self.current_frame = 0

    def get_metadata(self):
        return {
            'frame_rate': self.frame_rate,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'total_frames': self.total_frames
        }

    def __iter__(self):
        return self

    def __next__(self):
        if self.video is None or not self.video.isOpened():
            raise StopVideoIteration
        if self.current_frame >= self.total_frames:
            self.video.release()
            raise StopVideoIteration
        ret, frame = self.video.read()
        if not ret:
            self.video.release()
            raise StopVideoIteration
        self.current_frame += 1

        # Apply preprocessing here, if any
        if self.preprocessing is not None:
            frame = self.preprocessing(frame)

        return frame

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()


class StopVideoIteration(Exception):
    pass


class StopProcessorIteration(Exception):
    pass


class VideoBatchCreator:
    def __init__(self, file_paths: List[str], preprocessing=None):
        self.file_paths = iter(file_paths)
        self.preprocessing = preprocessing
        self.current_video_processor = None
        self.finished_paths = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if not self.current_video_processor:  # if no current video processor, create one
                try:
                    video_path = next(self.file_paths)
                    self.current_video_processor = VideoProcessor(video_path, preprocessing=self.preprocessing)
                except StopIteration:  # if no more file_paths
                    self.finished_paths = True
                    raise StopProcessorIteration("No more videos to process.")
                except FileNotFoundError as e:
                    print(e)
                    continue
            try:
                return next(self.current_video_processor)
            except StopIteration:  # finished processing current video processor, move to the next one
                self.current_video_processor = None
                raise StopVideoIteration("Moving to the next video.")


class SequentialBatchVideoProcessor:
    def __init__(self, file_paths: List[str], batch_size: int, frames_per_batch: int, shape: tuple, preprocessing=None,
                 padding='post', padding_frame=None):
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.frames_per_batch = frames_per_batch
        self.frame_height, self.frame_width, self.color_channels = shape
        if padding_frame is None:
            self.padding_frame = np.zeros((self.frame_height, self.frame_width, self.color_channels), dtype=np.uint8)

        self.video_creators = [(False, VideoBatchCreator(file_paths[i::batch_size], preprocessing)) for i in
                               range(batch_size)]

    def __iter__(self):
        return self

    def __next__(self):
        batch = [[] for _ in range(self.batch_size)]
        batch_all_finished = True

        for i, (finished, creator) in enumerate(self.video_creators):
            if not finished:
                frames_fetched = 0
                batch_all_finished = False
                try:
                    while frames_fetched < self.frames_per_batch:
                        frame = next(creator)
                        batch[i].append(frame)
                        frames_fetched += 1
                except StopVideoIteration:
                    # Mark the creator as finished for the current video
                    self.video_creators[i] = (True, creator)
                    print(f"Video {i} finished.")
                except StopProcessorIteration:
                    print(f"Video Batch: {i} finished.")

            while len(batch[i]) < self.frames_per_batch:
                batch[i].append(self.padding_frame)

        if batch_all_finished:
            completed = True
            for i in range(len(self.video_creators)):
                self.video_creators[i] = (self.video_creators[i][1].finished_paths, self.video_creators[i][1])
                if not self.video_creators[i][0] or completed is False:
                    print(f"Video Batch: {i} moving to next Path.")
                    completed = False

            if completed:
                raise StopIteration("All videos processed.")
            else:
                return self.__next__()

        # Flatten the batch and reshape it to the desired dimensions
        flat_batch = [frame for sublist in batch for frame in sublist]
        return np.array(flat_batch).reshape(self.batch_size, self.frames_per_batch, self.frame_height, self.frame_width,
                                            self.color_channels)


class ParallelBatchVideoProcessor:

    def __init__(self, file_paths_labels: List[tuple[str, Any]], batch_size: int, frames_per_batch: int, shape: tuple, preprocessing=None,
                 padding='post', padding_frame=None):
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.frames_per_batch = frames_per_batch
        self.frame_height, self.frame_width, self.color_channels = shape
        if padding_frame is None:
            self.padding_frame = np.zeros((self.frame_height, self.frame_width, self.color_channels), dtype=np.uint8)

        # Distribute video paths evenly among VideoBatchCreator instances
        self.video_paths_iter = iter(file_paths_labels)
        self.paths_completed = len(file_paths_labels) == 0
        self.new_video_flags = [True for _ in range(batch_size)]

        def videoProcessorGenerator():
            while True:
                try:
                    video_path, labels = next(self.video_paths_iter)
                    yield VideoProcessor(video_path, preprocessing=self.preprocessing), labels
                except StopIteration:
                    raise StopIteration("No more videos to process.")
                except FileNotFoundError as e:
                    print(e)
                    continue

        self.video_generator = videoProcessorGenerator()
        self.video_processors = [next(self.video_generator) for _ in range(batch_size)]

    def __iter__(self):
        return self

    def __next__(self):
        batch_frames = [[] for _ in range(self.batch_size)]
        batch_labels = [[] for _ in range(self.batch_size)]
        batch_new_vid_indicator = [[] for _ in range(self.batch_size)]
        batch_all_finished = True

        for i, (video_processor, label) in enumerate(self.video_processors):
            frames_fetched = 0
            try:
                while frames_fetched < self.frames_per_batch:
                    try:
                        frame = next(video_processor)
                    except StopVideoIteration:
                        if not self.paths_completed:
                            self.video_processors[i] = None
                            self.video_processors[i] = next(self.video_generator)
                            video_processor = self.video_processors[i][0]
                            label = self.video_processors[i][1]
                            self.new_video_flags[i] = True
                            # Skip to the next iteration of the while loop to continue fetching frames
                            continue
                        else:
                            # If no more videos are left to process, break out of the while loop
                            break
                    batch_all_finished = False
                    batch_frames[i].append(frame)
                    batch_labels[i].append(label)
                    batch_new_vid_indicator[i].append(1 if self.new_video_flags[i] else 0)
                    self.new_video_flags[i] = False
                    frames_fetched += 1
            except StopIteration:
                self.paths_completed = True
                print(f"Video {i} finished.")

            # Pad the current slot if it's not fully populated
            while len(batch_frames[i]) < self.frames_per_batch:
                batch_frames[i].append(self.padding_frame)
                batch_labels[i].append(np.zeros_like(label))
                batch_new_vid_indicator[i].append(0)

        if batch_all_finished:
            if self.paths_completed:
                raise StopIteration("All videos processed.")
            else:
                return self.__next__()

        flat_batch_frames = [frame for sublist in batch_frames for frame in sublist]
        batch_frames_np = np.array(flat_batch_frames).reshape(self.batch_size, self.frames_per_batch, self.frame_height,
                                                              self.frame_width, self.color_channels)

        return batch_frames_np, np.array(batch_labels), np.array(batch_new_vid_indicator)


# class _ContinuousBatchVideoProcessor:
#
#     def __init__(self, file_paths_labels: List[tuple[str, Any]], batch_size: int, frames_per_batch: int, shape: tuple, preprocessing=None,
#                  padding='post', padding_frame=None):
#         self.batch_size = batch_size
#         self.preprocessing = preprocessing
#         self.frames_per_batch = frames_per_batch
#         self.frame_height, self.frame_width, self.color_channels = shape
#         if padding_frame is None:
#             self.padding_frame = np.zeros((self.frame_height, self.frame_width, self.color_channels), dtype=np.uint8)
#
#         self.video_paths_iter = iter(file_paths_labels)
#         self.paths_completed = len(file_paths_labels) == 0
#         self.finished_current_video = False
#
#         def videoProcessorGenerator():
#             while True:
#                 try:
#                     video_path, labels = next(self.video_paths_iter)
#                     yield VideoProcessor(video_path, preprocessing=self.preprocessing), labels
#                 except StopIteration:
#                     self.paths_completed = True
#                     yield None, None
#                 except FileNotFoundError as e:
#                     print(e)
#                     continue
#
#         self.video_generator = videoProcessorGenerator()
#         self.current_video_processor, self.current_labels = next(self.video_generator)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.paths_completed and self.finished_current_video:
#             raise StopIteration("All videos processed.")
#
#         batch_frames = [[] for _ in range(self.batch_size)]
#         batch_labels = [None] * self.batch_size
#         new_video_flags = [0 for _ in range(self.batch_size)]
#
#         for i in range(self.batch_size):
#             frames_fetched = 0
#             while frames_fetched < self.frames_per_batch:
#                 if self.finished_current_video:
#                     frame = self.padding_frame
#                     new_video_flags[i] = 1
#                 else:
#                     try:
#                         frame = next(self.current_video_processor)
#                     except StopVideoIteration:
#                         self.finished_current_video = True
#                         continue
#
#                 batch_frames[i].append(frame)
#                 batch_labels[i] = self.current_labels  # Assign the single label for the batch item here
#                 frames_fetched += 1
#
#             if self.finished_current_video and not self.paths_completed:
#                 try:
#                     self.current_video_processor, self.current_labels = next(self.video_generator)
#                     self.finished_current_video = False
#                 except StopIteration:
#                     self.paths_completed = True
#
#         flat_batch_frames = [frame for sublist in batch_frames for frame in sublist]
#         batch_frames_np = np.array(flat_batch_frames).reshape(self.batch_size, self.frames_per_batch, self.frame_height,
#                                                               self.frame_width, self.color_channels)
#
#         return batch_frames_np, np.array(batch_labels), np.array(new_video_flags)


class ContinuousBatchVideoProcessor:

    def __init__(self, file_paths_labels: List[tuple[str, Any]], batch_size: int, frames_per_batch: int, shape: tuple, preprocessing=None,
                 padding='post', padding_frame=None, training=True):
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.frames_per_batch = frames_per_batch
        self.frame_height, self.frame_width, self.color_channels = shape
        if padding_frame is None:
            self.padding_frame = np.zeros((self.frame_height, self.frame_width, self.color_channels), dtype=np.uint8)

        self.video_paths_iter = iter(file_paths_labels)
        self.paths_completed = len(file_paths_labels) == 0
        self.finished_current_video = False
        self.training = training
        self.randomize_augment_parameters()

        def videoProcessorGenerator():
            while True:
                try:
                    video_path, labels = next(self.video_paths_iter)
                    yield VideoProcessor(video_path, preprocessing=self.preprocessing), labels
                except StopIteration:
                    self.paths_completed = True
                    yield None, None
                except FileNotFoundError as e:
                    print(e)
                    continue

        self.video_generator = videoProcessorGenerator()
        self.current_video_processor, self.current_labels = next(self.video_generator)
        self.new_video_next_batch = False  # Flag to indicate the start of a new video in the next batch

    def __iter__(self):
        return self

    def randomize_augment_parameters(self):
        self.flip = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) == 1
        self.delta = tf.random.uniform(shape=[], minval=-0.1, maxval=0.1)
        self.contrast_factor = tf.random.uniform(shape=[], minval=0.1, maxval=0.2)
        self.rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

    def apply_augmentation(self, frame):
        if self.flip:
            frame = tf.image.flip_left_right(frame)
        frame = tf.image.adjust_brightness(frame, self.delta)
        frame = tf.image.adjust_contrast(frame, self.contrast_factor)
        frame = tf.image.rot90(frame, k=self.rotation)
        return frame


    def __next__(self):
        if self.paths_completed:
            raise StopIteration("All videos processed.")

        batch_frames = [[] for _ in range(self.batch_size)]
        batch_labels = [None for _ in range(self.batch_size)]
        new_video_flags = [0 for _ in range(self.batch_size)]

        # Set the flag for the first item of the batch if a new video is starting
        if self.new_video_next_batch:
            new_video_flags[0] = 1
            self.new_video_next_batch = False
            self.randomize_augment_parameters()

        for i in range(self.batch_size):
            frames_fetched = 0
            while frames_fetched < self.frames_per_batch:
                if self.finished_current_video:
                    frame = self.padding_frame
                    label = np.zeros_like(self.current_labels)
                else:
                    try:
                        frame = next(self.current_video_processor)
                        if self.training:
                            self.apply_augmentation(frame)
                        label = self.current_labels
                    except StopVideoIteration:
                        self.finished_current_video = True
                        frame = self.padding_frame
                        label = np.zeros_like(self.current_labels)
                        # Indicate that the next batch should start with a new video
                        self.new_video_next_batch = True

                batch_frames[i].append(frame)
                if batch_labels[i] is None:
                    batch_labels[i] = label
                frames_fetched += 1

        # If the current video is finished, prepare to start a new video in the next batch
        if self.finished_current_video and not self.paths_completed:
            try:
                self.current_video_processor, self.current_labels = next(self.video_generator)
                self.finished_current_video = False
            except StopIteration:
                self.paths_completed = True

        flat_batch_frames = [frame for sublist in batch_frames for frame in sublist]
        batch_frames_np = np.array(flat_batch_frames).reshape(self.batch_size, self.frames_per_batch, self.frame_height,
                                                              self.frame_width, self.color_channels)

        return batch_frames_np, np.array(batch_labels), np.array(new_video_flags)


padded_frame = np.zeros((240, 320, 3), dtype=np.uint8)


def display_batch(batch):
    num_videos = len(batch)
    grid_size = int(np.ceil(np.sqrt(num_videos)))  # Determine grid size (e.g., for 4 videos, grid_size will be 2)
    frame_height, frame_width = 240, 320

    for i in range(len(batch[0])):
        grid_frames = []

        for video in batch:
            frame = video[i]
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            grid_frames.append(resized_frame)

        while len(grid_frames) < grid_size ** 2:
            grid_frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

        # Create the grid
        rows = []
        for row in range(grid_size):
            start = row * grid_size
            end = start + grid_size
            rows.append(np.hstack(grid_frames[start:end]))

        grid = np.vstack(rows)

        cv2.imshow('Batch Videos', grid)
        key = cv2.waitKey(100)
        if key == 27:  # Exit loop if ESC is pressed
            break


if __name__ == "__main__":
    from dataset.preprocess_datasets.testing_generators import get_data_for_loading

    video_files = get_data_for_loading([1])[0]
    # videos = [file for file, _ in video_files]
    batch_size = 4
    frames_per_batch = 120
    frame_shape = (240, 320, 3)  # Height, width, channels
    video_processor = ContinuousBatchVideoProcessor(video_files, batch_size, frames_per_batch, frame_shape)
    for batch, labels, indicator in video_processor:
        print(batch.shape)
        print(np.argmax(labels, axis=1))
        print(indicator)
        display_batch(batch)
    cv2.destroyAllWindows()

    print("All videos processed.")
