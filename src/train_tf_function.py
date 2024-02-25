import cv2
import numpy as np
from keras.src.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K
import datetime
import random
import tensorflow as tf
from pathlib import Path
from typing import List
import pandas as pd
from queue import Queue
from threading import Thread

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard

from dataset.preprocess_datasets.data_iterator import get_train_test_size
from models.ADAI import ActionDetectionModel, SpatialAttentionLayer, TemporalAttentionLayer
from tensorflow.keras.utils import custom_object_scope

# updated hyperparameters
best_hyperparameters = {'lstm_units': 512, 'dense_units': 1024, 'dropout_rate': 0.5, 'fine_tune_at': 4,
                        'num_frames': 25, 'tuner/epochs': 4, 'tuner/initial_epoch': 2, 'tuner/bracket': 2,
                        'tuner/round': 1, 'tuner/trial_id': '0005'}

SEQUENCE_LENGTH = best_hyperparameters['num_frames']
BATCH_SIZE = 1
EPOCHS = 100
MODEL_SAVE_PATH = '_model_checkpoint.keras'
DROP_RATE = best_hyperparameters['dropout_rate']
TRAINING_DATASET = [1, 2, 3]
# VALIDATION_DATASET = [3]

from dataset.preprocess_datasets.testing_generators import get_data_for_loading
from dataset.preprocess_datasets.video_handling import ContinuousVideoProcessor, VideoProcessor

train_paths, test_paths = get_data_for_loading(TRAINING_DATASET)
# _, val_path = get_data_for_loading(VALIDATION_DATASET)
train_size, test_size = get_train_test_size(TRAINING_DATASET)
# _, val_size = get_train_test_size(VALIDATION_DATASET)


def calculate_video_accuracy(predictions_accumulated, labels_accumulated):
    all_predictions = np.concatenate(predictions_accumulated, axis=0)
    all_labels = np.concatenate(labels_accumulated, axis=0)
    correct_predictions = np.sum(np.argmax(all_predictions, axis=0) == np.argmax(all_labels, axis=0))
    total_items = len(all_labels)
    return correct_predictions / total_items if total_items > 0 else 0


import threading
import functools


class VideoProcessor:
    def __init__(self, file_path, preprocessing=None, target_shape=None, training=False):
        self.video = None
        self.file_path = file_path
        self.preprocessing = preprocessing
        self.target_shape = target_shape
        self.training = training
        self.lock = threading.Lock()
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
                frame = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]))
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
        try:
            self.video = cv2.VideoCapture(self.file_path)
            if not self.video.isOpened():
                raise ValueError("Could not open video file.")
            # Extract metadata
            if self.training:
                self.randomize_augment_parameters()
            self.original_frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        except cv2.error as e:
            raise FileNotFoundError(f"{e}: Could not open video file: {self.file_path}")
        frame_interval = self.original_frame_rate / fps_target  # Calculate the interval between frames to fetch

        while True:  # The generator will continue until it can't yield a full sequence
            sequence = []
            # with self.lock:
            start_frame_index = self.current_frame
            self.current_frame = int(round(start_frame_index + num_frames * frame_interval))  # Update current_frame

            for i in range(num_frames):
                target_frame_index = int(round(start_frame_index + i * frame_interval))
                if target_frame_index >= self.total_frames:
                    self.video.release()
                    return  # Stop the generator if there aren't enough frames left

                self.video.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                ret, frame = self.video.read()
                if not ret:
                    self.video.release()
                    return  # Stop the generator if unable to read the frame

                frame = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]))

                # convert frame to tensor
                frame = tf.convert_to_tensor(frame, dtype=tf.float32)

                if self.preprocessing is not None:
                    frame = self.preprocessing(frame)
                if self.training:
                    frame = self.apply_augmentation(frame)
                sequence.append(frame)

            # Stack the sequence and remove the first singleton dimension (if present)
            sequence_tensor = tf.stack(sequence)
            if sequence_tensor.shape[0] == 1:  # Check if the first dimension is singleton
                sequence_tensor = tf.squeeze(sequence_tensor, axis=0)

            # yield tf.expand_dims(sequence_tensor, axis=0)
            yield sequence_tensor

    def standardize_batch_fetch_generator(self, fps_target, num_frames, batch_size = 1):
        try:
            self.video = cv2.VideoCapture(self.file_path)
            if not self.video.isOpened():
                raise ValueError("Could not open video file.")
            # Extract metadata
            if self.training:
                self.randomize_augment_parameters()
            self.original_frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        except cv2.error as e:
            raise FileNotFoundError(f"{e}: Could not open video file: {self.file_path}")
        frame_interval = self.original_frame_rate / fps_target  # Calculate the interval between frames to fetch

        while True:  # The generator will continue until it can't yield a full sequence
            sequence = []
            with self.lock:
                start_frame_index = self.current_frame
                self.current_frame = int(round(start_frame_index + num_frames * frame_interval))  # Update current_frame

                for i in range(num_frames):
                    target_frame_index = int(round(start_frame_index + i * frame_interval))
                    if target_frame_index >= self.total_frames:
                        self.video.release()
                        return  # Stop the generator if there aren't enough frames left

                    self.video.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                    ret, frame = self.video.read()
                    if not ret:
                        self.video.release()
                        return  # Stop the generator if unable to read the frame

                    # convert frame to tensor
                    frame = tf.convert_to_tensor(frame, dtype=tf.float32)

                    if self.preprocessing is not None:
                        frame = self.preprocessing(frame)
                    if self.training:
                        self.randomize_augment_parameters(True)
                        frame = self.apply_augmentation(frame)
                    sequence.append(frame)

            # Stack the sequence and remove the first singleton dimension (if present)
            sequence_tensor = tf.stack(sequence)
            if sequence_tensor.shape[0] == 1:  # Check if the first dimension is singleton
                sequence_tensor = tf.squeeze(sequence_tensor, axis=0)

            yield tf.expand_dims(sequence_tensor, axis=0)

    # def randomize_augment_parameters(self, post_setup=False):
    #     if not post_setup:
    #         self.flip = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32).numpy() == 1
    #     self.delta = tf.random.uniform(shape=[], minval=-0.05,
    #                                    maxval=0.05).numpy()  # More conservative range for brightness
    #     self.contrast_factor = tf.random.uniform(shape=[], minval=0.9,
    #                                              maxval=1.1).numpy()  # Narrower range for contrast
    #     self.to_grayscale = tf.random.uniform(shape=[], minval=0, maxval=2,
    #                                           dtype=tf.int32).numpy() == 1  # Randomize grayscale conversion

    def randomize_augment_parameters(self, post_setup=False):
        if not post_setup:
            # Generate a random boolean tensor for flipping
            self.flip = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) < 1
        self.delta = tf.random.uniform(shape=[], minval=-0.05, maxval=0.05)  # Brightness
        self.contrast_factor = tf.random.uniform(shape=[], minval=0.9, maxval=1.1)  # Contrast
        # Generate a random boolean tensor for grayscale conversion
        self.to_grayscale = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) < 1

    def apply_augmentation(self, frame_tensor):
        # Normalize the frame tensor
        frame_tensor = tf.cast(frame_tensor, tf.float32) / 255.0

        # Conditionally flip the image horizontally
        frame_tensor = tf.cond(self.flip,
                               lambda: tf.image.flip_left_right(frame_tensor),
                               lambda: frame_tensor)

        # Adjust brightness and contrast
        frame_tensor = tf.image.adjust_brightness(frame_tensor, self.delta)
        frame_tensor = tf.image.adjust_contrast(frame_tensor, self.contrast_factor)

        # Conditionally convert to grayscale and then back to RGB

        frame_tensor = tf.cond(self.to_grayscale,
                               lambda: self.to_grayscale_and_back(frame_tensor),
                               lambda: frame_tensor)

        # Clip values to ensure they are within [0, 1]
        frame_tensor = tf.clip_by_value(frame_tensor, 0.0, 1.0)

        return frame_tensor

    def to_grayscale_and_back(self, x):
        grayscale = tf.image.rgb_to_grayscale(x)
        return tf.image.grayscale_to_rgb(grayscale)

    # def apply_augmentation(self, frame_tensor):
    #     # Normalize the frame tensor
    #     frame_tensor = tf.cast(frame_tensor, tf.float32) / 255.0
    #
    #     # Conditionally flip the image horizontally
    #     frame_tensor = tf.cond(self.flip > 0,
    #                            lambda: tf.image.flip_left_right(frame_tensor),
    #                            lambda: frame_tensor)
    #
    #     # Adjust brightness and contrast
    #     frame_tensor = tf.image.adjust_brightness(frame_tensor, self.delta)
    #     frame_tensor = tf.image.adjust_contrast(frame_tensor, self.contrast_factor)
    #
    #     # Conditionally convert to grayscale and then back to RGB
    #
    #     frame_tensor = tf.cond(self.to_grayscale > 0,
    #                            lambda: self.to_grayscale_and_back(frame_tensor),
    #                            lambda: frame_tensor)
    #
    #     # Clip values to ensure they are within [0, 1]
    #     frame_tensor = tf.clip_by_value(frame_tensor, 0.0, 1.0)
    #
    #     return frame_tensor

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()

    def number_remaining_frames(self):
        return self.total_frames - self.current_frame




@tf.function
def train_step(model, sequence_set, label, loss_fn, optimizer, train_sequence_metrics, train_video_metrics, sequences_read, accumulated_loss,
               accumulated_gradients, accumulated_predictions):
    sequences_read.assign(0)
    accumulated_loss.assign(0.0)
    averaged_predictions = tf.zeros_like(accumulated_predictions)

    # Initialize aggregated predictions tensor
    accumulated_predictions.assign(tf.zeros_like(accumulated_predictions))

    for ag in accumulated_gradients:
        ag.assign(tf.zeros_like(ag))

    for sequence in sequence_set:
        sequences_read.assign_add(1)
        with tf.GradientTape() as tape:
            prediction = model(sequence, training=True)
            accumulated_predictions.assign_add(prediction)
            averaged_predictions = accumulated_predictions / tf.cast(sequences_read, tf.float32)
            sequence_loss = loss_fn(label, averaged_predictions)

        # Aggregate predictions by summing
        gradients = tape.gradient(sequence_loss, model.trainable_variables)
        for i, (ag, g) in enumerate(zip(accumulated_gradients, gradients)):
            if g is not None:
                accumulated_gradients[i].assign_add(g)

        for metric in train_sequence_metrics:
            metric.update_state(label, tf.one_hot(tf.argmax(prediction, axis=-1), depth=101))

        accumulated_loss.assign_add(sequence_loss)

    # Use the average prediction for metrics (or other purposes)
    for metric in train_video_metrics:
        metric.update_state(label, tf.one_hot(tf.argmax(averaged_predictions, axis=-1), depth=101))

    # Apply gradients
    avg_gradients = [ag / tf.cast(sequences_read, tf.float32) for ag in accumulated_gradients]
    optimizer.apply_gradients(zip(avg_gradients, model.trainable_variables))

    model.reset_states()

    return accumulated_loss, sequences_read, averaged_predictions

def create_video_dataset(video_path, label, frame_height, frame_width, channels, sequence_length):
    # Create a dataset for a single video using from_generator
    video_ds = tf.data.Dataset.from_generator(
        lambda vp=VideoProcessor(video_path, preprocess_input, (frame_height, frame_width, channels),
                                 False): vp.standardize_fetch_generator(25, sequence_length),
        output_signature=tf.TensorSpec(shape=(sequence_length, frame_height, frame_width, channels), dtype=tf.float32)
    )

    # Apply transformations
    video_ds = video_ds.map(lambda x: tf.expand_dims(x, axis=0), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch to improve efficiency
    video_ds = video_ds.prefetch(tf.data.AUTOTUNE)

    # Expand the label dimensions to match the video data
    label_ds = tf.data.Dataset.from_tensors(tf.expand_dims(label, axis=0))

    # Zip the video dataset with the label dataset
    video_label_ds = tf.data.Dataset.zip((video_ds, label_ds))

    return video_label_ds

def print_summary_line(video_read, portion_size, loss_accumulated, sequence_read, sequence_metrics):
    sequence_metrics_values = {metric.name: metric.result().numpy() for metric in sequence_metrics}
    percentage = video_read / portion_size * 100
    summary_format = '\rCompleted: {}/{}, ({:.2f}%). Loss: {:.4f}\tSequence Metrics: {}' #\tOn video path: {}'

    average_loss = loss_accumulated / tf.cast(sequence_read, tf.float32)

    sequence_metrics_str = ', '.join(
        f'{name}: {value:.4f}' for name, value in sequence_metrics_values.items())

    # Format the summary line with actual values
    summary_line = summary_format.format(video_read, portion_size, percentage,
                                         average_loss,
                                         sequence_metrics_str) #, video_path)

    # Print the summary line, overwriting the previous one
    print(summary_line, end='', flush=True)

def print_summary_line_threaded(video_read, portion_size, loss_accumulated, sequence_read, sequence_metrics):
    global summary_thread
    summary_thread = None

    # Only start a new thread if the previous one has finished
    if summary_thread is None or not summary_thread.is_alive():
        summary_thread = threading.Thread(target=print_summary_line, args=(
        video_read, portion_size, loss_accumulated, sequence_read, sequence_metrics))
        summary_thread.start()

def train_model(load_previous_model=False):
    # Model parameters
    num_frames, frame_width, frame_height, channels, num_classes = SEQUENCE_LENGTH, 320, 240, 3, 101
    lstm_units = best_hyperparameters['lstm_units']
    dense_units = best_hyperparameters['dense_units']

    def create_video_dataset(video_path, label, processor, sequence_length):
        video_ds = processor.standardize_fetch_generator(25, sequence_length)
        label_ds = tf.expand_dims(tf.cast(label, tf.float32), axis=0)
        return video_ds, label_ds, video_path

    if load_previous_model:
        model = load_model(MODEL_SAVE_PATH, custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer,
                                                            'TemporalAttentionLayer': TemporalAttentionLayer})
    else:
        model = ActionDetectionModel(1, num_frames, frame_width, frame_height, channels,
                                     num_classes, lstm_units, dense_units,
                                     dropout_rate=DROP_RATE,
                                     fine_tune_until=best_hyperparameters['fine_tune_at'])

    initial_learning_rate = 0.05
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4, beta_1=0.9, beta_2=0.999,
                      epsilon=1e-07, amsgrad=False, name='AdamW')

    log_dir = "logs/fit_with_AdamW/"

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min', initial_value_threshold=5.12366),
        TensorBoard(log_dir=log_dir + "ActionDetectionModel", histogram_freq=1,
                    update_freq='batch'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    def randomize_dataset_threads(video_datasets):
        global randomize_thread
        randomize_thread = None

        if randomize_thread is None or not randomize_thread.is_alive():
            randomize_thread = threading.Thread(target=random.shuffle, args=(video_datasets,))
            randomize_thread.start()

    # metrics

    # Compile the model with learning rate schedule
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()

    # I am not sure if this is needed. I think it is not needed
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=['accuracy', 'categorical_accuracy', 'recall', 'precision'])

    # create metrics per video
    # train_video_metrics = [tf.keras.metrics.CategoricalAccuracy(name='video_accuracy')]
    # val_video_metrics = [tf.keras.metrics.CategoricalAccuracy(name='video_accuracy')]
    train_sequence_metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                              tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                              tf.keras.metrics.Recall(name='recall'),
                              tf.keras.metrics.Precision(name='precision')]
    val_sequence_metrics = [tf.keras.metrics.CategoricalAccuracy(name='val_accuracy'),
                            tf.keras.metrics.CategoricalAccuracy(name='val_categorical_accuracy'),
                            tf.keras.metrics.Recall(name='val_recall'),
                            tf.keras.metrics.Precision(name='val_precision')]

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    train_paths_sample = train_paths[:len(train_paths)]
    val_path_sample = test_paths[:len(test_paths)]

    train_video_datasets = [
        (
            tf.data.Dataset.from_generator(
                lambda vp=VideoProcessor(video_path, preprocess_input, (frame_height, frame_width, channels),
                                         True): vp.standardize_fetch_generator(25, SEQUENCE_LENGTH),
                # Use a lambda to capture the current VideoProcessor instance
                output_signature=tf.TensorSpec(shape=(SEQUENCE_LENGTH, frame_height, frame_width, channels),
                                               dtype=tf.float32)
            )
            .map(lambda x: tf.expand_dims(x, axis=0),
                 num_parallel_calls=tf.data.AUTOTUNE)  # Apply any required transformations here
            .prefetch(tf.data.AUTOTUNE),  # Prefetching added here
            tf.expand_dims(label, axis=0),
            video_path
        )
        for video_path, label in train_paths_sample
    ]
    test_video_datasets = [
        (
            tf.data.Dataset.from_generator(
                lambda vp=VideoProcessor(video_path, preprocess_input, (frame_height, frame_width, channels),
                                         False): vp.standardize_fetch_generator(25, SEQUENCE_LENGTH),
                # Use a lambda to capture the current VideoProcessor instance
                output_signature=tf.TensorSpec(shape=(SEQUENCE_LENGTH, frame_height, frame_width, channels),
                                               dtype=tf.float32)
            )
            .map(lambda x: tf.expand_dims(x, axis=0),
                 num_parallel_calls=tf.data.AUTOTUNE)  # Apply any required transformations here
            .prefetch(tf.data.AUTOTUNE),  # Prefetching added here
            tf.expand_dims(label, axis=0),
            video_path
        )
        for video_path, label in val_path_sample
    ]

    accumulative_loss_tensor = tf.Variable(0.0, dtype=tf.float32)
    accumulative_gradients_tensor = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32)) for tv in model.trainable_variables]
    sequences_read_tensor = tf.Variable(0, dtype=tf.int32)
    accumulated_predictions_tensor = tf.Variable(tf.zeros((1, num_classes), dtype=tf.float32))

    print('\a')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        for metrics in [train_sequence_metrics, val_sequence_metrics]:
            for metric in metrics:
                metric.reset_states()

        logs = {'epoch': epoch + 1}
        for callback in callbacks:
            callback.on_epoch_begin(epoch, logs=logs)

        # random sample of the training dataset
        train_portion = random.sample(train_video_datasets, len(train_video_datasets)//20)

        videos_read = 0
        sequences_read = 0
        train_portion_size = len(train_portion)
        accumulated_loss = 0


        # every video in the training dataset
        for video_sequence_set, label, video_path in train_portion:
            accumulative_loss_tensor.assign(0.0)
            accumulative_gradients_tensor = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32)) for tv in model.trainable_variables]
            sequences_read_tensor.assign(0)
            # Iterate over the dataset
            loss, sequence_read, ave_predictions = train_step(model, video_sequence_set, label, loss_fn, optimizer,
                                              train_sequence_metrics, sequences_read_tensor, accumulative_loss_tensor,
                                              accumulative_gradients_tensor, accumulated_predictions_tensor)
            videos_read += 1
            sequences_read += sequence_read
            accumulated_loss += loss
            train_loss_metric.update_state(accumulated_loss)
            print_summary_line_threaded(videos_read, train_portion_size, train_loss_metric, sequences_read,
                                        train_sequence_metrics)


        print()

        val_video_read = 0
        # sample of the validation dataset
        val_portion = random.sample(test_video_datasets, len(test_video_datasets)//20)
        val_portion_size = len(val_portion)

        val_loss_accumulated = 0
        val_sequence_read = 0

        # every video in the training dataset
        for video_sequence_set, label, video_path in val_portion:
            val_sequences_read = 0
            accumulated_loss = 0
            accumulative_gradients_tensor = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32)) for tv in model.trainable_variables]
            averaged_predictions = tf.zeros_like(accumulated_predictions_tensor)
            accumulative_gradients_tensor = [tf.Variable(tf.zeros_like(tv, dtype=tf.float32)) for tv in model.trainable_variables]

            for ag in accumulative_gradients_tensor:
                ag.assign(tf.zeros_like(ag))

            for sequence in video_sequence_set:
                val_sequences_read += 1
                prediction = model(sequence, training=False)
                accumulated_predictions_tensor.assign_add(prediction)
                averaged_predictions = accumulated_predictions_tensor / tf.cast(val_sequence_read, tf.float32)
                sequence_loss = loss_fn(label, averaged_predictions)

                val_loss_accumulated += sequence_loss
                val_loss_metric.update_metric(val_loss_accumulated)

            for metric in val_sequence_metrics:
                metric.update_state(label, tf.one_hot(tf.argmax(averaged_predictions, axis=-1), depth=101))

            val_loss_accumulated += accumulated_loss
            val_loss_metric.update_state(accumulated_loss)
            print(f'val_loss: {val_loss_metric.result().numpy()}')

            model.reset_states()
            val_video_read += 1

            print_summary_line_threaded(val_video_read, val_portion_size, val_loss_accumulated, val_sequences_read,
                                        val_sequence_metrics)

        logs = {
            'loss': train_loss_metric.result().numpy(),
            'val_loss': val_loss_metric.result().numpy(),
            'lr': lr_schedule(epoch),
        }
        logs.update({f'train_{metric.name}': metric.result().numpy() for metric in train_sequence_metrics})
        logs.update({f'val_{metric.name}': metric.result().numpy() for metric in val_sequence_metrics})
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    for callback in callbacks:
        callback.on_train_end()

    return model

if __name__ == "__main__":
    train_model(load_previous_model=True)
