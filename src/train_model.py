import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
from keras.src.applications.vgg19 import preprocess_input
from tensorflow import Tensor
import queue
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random
import tensorflow as tf
import keras_cv
import concurrent.futures

from tensorflow.keras.callbacks import TensorBoard

from dataset.preprocess_datasets.data_iterator import get_train_test_size
from models.ADAI import ActionDetectionModel, SpatialAttentionLayer, TemporalAttentionLayer
from dataset.preprocess_datasets.testing_generators import get_data_for_loading

best_hyperparameters = {'lstm_units': 1024, 'dense_units': 256, 'dropout_rate': 0.5, 'fine_tune_at': 4,
                        'num_frames': 25}

SEQUENCE_LENGTH = best_hyperparameters['num_frames']
BATCH_SIZE = 1
EPOCHS = 100
MODEL_SAVE_PATH = '../models/previous models/model_trained_by_sequence.keras'
DROP_RATE = best_hyperparameters['dropout_rate']
TRAINING_DATASET = [1, 2, 3]

train_paths, test_paths = get_data_for_loading(TRAINING_DATASET)
train_size, test_size = get_train_test_size(TRAINING_DATASET)


class VideoProcessor:
    def __init__(self, file_path: str, target_shape: Optional[Tuple[int, int, int]] = (240, 320, 3),
                 frame_buffer_size: int = 200, training: bool = True):
        """
        Initializes the object with the given file path and target shape.

        :param file_path: The path of the video file.
        :param target_shape: The desired shape of the video frames as a tuple of width and height.
        """
        self.video = None
        self.original_frame_rate = None
        self.frame_width = None
        self.frame_height = None
        self.total_frames = None
        self.file_path = file_path
        self.target_shape = target_shape
        self.current_frame = 0
        self.training = training

        self.should_flip: tf.Tensor = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) < 1
        # Define the RandomRotation layer with a small range (-10 to 10 degrees)
        # Degrees are specified in fractions of 2π, so 10 degrees is about 10/360 = 0.0277
        self.rotation_layer: keras_cv.layers.RandomRotation = keras_cv.layers.RandomRotation(factor=(-0.0277, 0.0277))
        self.should_grayscale: tf.Tensor = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) < 1
        try:
            self.video = cv2.VideoCapture(self.file_path)
            if not self.video.isOpened():
                raise ValueError("Could not open video file.")
            self.original_frame_rate = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        except cv2.error as e:
            raise FileNotFoundError(f"{e}: Could not open video file: {self.file_path}")
        self.fps_target = 25
        self.frame_buffer = Queue(maxsize=frame_buffer_size)
        self.prefetch_thread = Thread(target=self.prefetch_frames, args=(self.fps_target,), daemon=True)
        self.prefetch_thread.start()

    def prefetch_frames(self, fps_target):
        """Function to continuously read, augment, and selectively store frames based on the target FPS."""
        skip_rate = int(round(self.original_frame_rate / fps_target))
        frame_count = 0

        while True:
            ret, frame = self.video.read()
            if not ret:
                self.frame_buffer.put(None)  # Signal that the video has ended
                break

            # Only add frames to the buffer based on the skip rate
            if frame_count % skip_rate == 0:
                frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
                if self.training:
                    frame_tensor = self.apply_augmentations(frame_tensor)

                frame_tensor = preprocess_input(frame_tensor)
                frame_tensor = tf.image.resize(frame_tensor, (self.target_shape[0], self.target_shape[1]))
                frame_tensor = frame_tensor / 255.0  # Normalize the pixel values
                self.frame_buffer.put(frame_tensor)

            frame_count += 1

    @tf.function
    def apply_augmentations(self, frame_tensor):
        # Check conditions outside the frame loop
        if self.should_flip:
            frame_tensor = tf.image.flip_left_right(frame_tensor)

        frame_tensor = self.rotation_layer(frame_tensor)  # Rotation is always applied

        # Other augmentations
        frame_tensor = tf.image.random_hue(frame_tensor, max_delta=0.08)
        frame_tensor = tf.image.random_saturation(frame_tensor, lower=0.7, upper=1.3)
        frame_tensor = tf.image.random_brightness(frame_tensor, max_delta=0.05)
        frame_tensor = tf.image.random_contrast(frame_tensor, lower=0.7, upper=1.3)
        frame_tensor = frame_tensor + tf.random.normal(tf.shape(frame_tensor), mean=0.0, stddev=0.1)

        if self.should_grayscale:
            frame_tensor = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(frame_tensor))

        return tf.clip_by_value(frame_tensor, 0.0, 1.0)

    def standardize_fetch_generator(self, num_frames):
        while True:
            sequence = []
            for _ in range(num_frames):
                frame = self.frame_buffer.get()  # Retrieve the next frame from the buffer
                if frame is None:  # End of video
                    self.video.release()
                    return

                sequence.append(frame)

            yield sequence

    def __del__(self):
        if self.video is not None and self.video.isOpened():
            self.video.release()

    def number_remaining_frames(self):
        return self.total_frames - self.current_frame


class VideoLoader:
    def __init__(self, video_list, max_queue_size=10, target_shape=(240, 320, 3), training=True):
        self.video_list = video_list
        self.queue = Queue(maxsize=max_queue_size)
        self.stop_signal = False
        self.target_shape = target_shape
        self.thread = None  # Initialize the thread reference
        self.training = training

    def video_loader_thread(self):
        for video_path, label in self.video_list:
            if self.stop_signal:
                break
            video_sequence, label = create_video_dataset(video_path, label, frame_shape=self.target_shape,
                                                         training=self.training)
            try:
                self.queue.put((video_sequence, label))  # Use a timeout for blocking
            except queue.Full:
                print("Queue is full, skipping...")
        self.stop_signal = True

    def start_loading(self):
        self.stop_signal = False
        self.thread = Thread(target=self.video_loader_thread)  # Assign the thread to the instance variable
        self.thread.start()

    def get_next_video(self):
        try:
            if not self.queue.empty() or not self.stop_signal:
                return self.queue.get(timeout=1)  # Use a timeout to avoid indefinite blocking
        except queue.Empty:
            print("Queue is empty...")
        return None

    def stop_loading(self):
        self.stop_signal = True
        if self.thread and self.thread.is_alive():  # Check if the thread has been started and is alive
            self.thread.join()  # Wait for the thread to finish

        # Optional: Clear the queue to ensure no residual data
        while not self.queue.empty():
            self.queue.get()


# Metrics
train_video_accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
train_video_recall = tf.keras.metrics.Recall(name='recall')
train_video_precision = tf.keras.metrics.Precision(name='precision')
train_video_f1_score = tf.keras.metrics.F1Score(name='f1_score', average='weighted')
train_seq_accuracy = tf.keras.metrics.CategoricalAccuracy(name='seq_accuracy')
train_seq_recall = tf.keras.metrics.Recall(name='seq_recall')
train_seq_precision = tf.keras.metrics.Precision(name='seq_precision')
train_seq_f1_score = tf.keras.metrics.F1Score(name='seq_f1_score', average='weighted')
val_video_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
val_video_recall = tf.keras.metrics.Recall(name='val_recall')
val_video_precision = tf.keras.metrics.Precision(name='val_precision')
val_video_f1_score = tf.keras.metrics.F1Score(name='val_f1_score', average='weighted')
val_seq_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_seq_accuracy')
val_seq_recall = tf.keras.metrics.Recall(name='val_seq_recall')
val_seq_precision = tf.keras.metrics.Precision(name='val_seq_precision')
val_seq_f1_score = tf.keras.metrics.F1Score(name='val_seq_f1_score', average='weighted')

# Loss metrics
train_video_loss_metric = tf.keras.metrics.Mean(name='loss')
val_video_loss_metric = tf.keras.metrics.Mean(name='val_loss')
train_sequence_loss_metric = tf.keras.metrics.Mean(name='seq_loss')
val_sequence_loss_metric = tf.keras.metrics.Mean(name='val_seq_loss')


# Loss function
def focal_loss(gamma=2., alpha=.25, from_logits=False):
    '''

    :param gamma: Exponent of the modulating factor (1 - p_t)^gamma
    :param alpha: Weight factor for the positive class
    :param from_logits: whether the input is a logit or a probability
    :return: Focal loss function
    '''
    def focal_loss_with_logits(logits, targets):
        y_pred = tf.sigmoid(logits)
        loss = targets * (-alpha * tf.pow((1 - y_pred), gamma) * tf.math.log(y_pred)) + \
               (1 - targets) * (-alpha * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred))
        return tf.reduce_sum(loss)

    def focal_loss_with_probs(probs, targets):
        '''
        References :Lee, J.-W., & Kang, H.-S. (2024). Three-Stage Deep Learning Framework for Video Surveillance. Applied Sciences (2076-3417), 14(1), 408. https://doi-org.lopes.idm.oclc.org/10.3390/app14010408

        :param probs: y_pred from the model (predicted probabilities)
        :param targets: y_true from the model (true labels)
        :return: Focal loss
        '''
        eps = 1e-7
        loss = targets * (-alpha * tf.pow((1 - probs), gamma) * tf.math.log(probs + eps)) + \
               (1 - targets) * (-alpha * tf.pow(probs, gamma) * tf.math.log(1 - probs + eps))
        return tf.reduce_sum(loss)

    return focal_loss_with_logits if from_logits else focal_loss_with_probs


# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_fn = focal_loss()

# Prediction
train_accumulated_predictions = tf.Variable(tf.zeros((1, 101), dtype=tf.float32))
val_accumulated_predictions = tf.Variable(tf.zeros((1, 101), dtype=tf.float32))

initial_learning_rate = 0.05
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
optimizer = AdamW(learning_rate=lr_schedule,
                  # weight_decay=1e-4, beta_1=0.9, beta_2=0.999,
                  # epsilon=1e-07,
                  amsgrad=False, name='AdamW')


@tf.function
def train_by_sequence_step(model, sequence_generator, label):
    global train_accumulated_predictions
    global train_video_loss_metric
    global train_sequence_loss_metric

    # load train metrics
    global train_video_accuracy
    global train_video_recall
    global train_video_precision
    global train_video_f1_score
    global train_seq_accuracy
    global train_seq_recall
    global train_seq_precision
    global train_seq_f1_score
    global loss_fn

    train_accumulated_predictions.assign(tf.zeros_like(train_accumulated_predictions))

    for sequence in sequence_generator:

        # Generate a random boolean tensor for grayscale conversion
        with tf.GradientTape() as tape:
            prediction = model(sequence, training=True)
            loss = loss_fn(label, prediction)

        train_sequence_loss_metric.update_state(loss)
        train_accumulated_predictions.assign_add(prediction)

        # Aggregate predictions by summing
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update train sequence metrics
        for metric in [train_seq_accuracy, train_seq_recall, train_seq_precision, train_seq_f1_score]:
            metric.update_state(label, prediction)

    # Update validation metrics
    for metric in [train_video_accuracy, train_video_recall, train_video_precision, train_video_f1_score]:
        metric.update_state(label, tf.nn.softmax(train_accumulated_predictions, axis=-1))

    train_video_loss_metric.update_state(train_sequence_loss_metric.result())

    model.reset_states()

    return


@tf.function
def val_by_sequence_step(model, sequence_generator, label):
    global val_accumulated_predictions
    global val_video_loss_metric
    global val_sequence_loss_metric

    # load train metrics
    global val_video_accuracy
    global val_video_recall
    global val_video_precision
    global val_video_f1_score
    global val_seq_accuracy
    global val_seq_recall
    global val_seq_precision
    global val_seq_f1_score
    global loss_fn

    val_accumulated_predictions.assign(tf.zeros_like(val_accumulated_predictions))

    for sequence in sequence_generator:
        # Generate a random boolean tensor for grayscale conversion
        with tf.GradientTape() as tape:
            prediction = model(sequence, training=False)
            loss = loss_fn(label, prediction)

        val_sequence_loss_metric.update_state(loss)
        val_accumulated_predictions.assign_add(prediction)

        # Aggregate predictions by summing
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update train sequence metrics
        for metric in [val_seq_accuracy, val_seq_recall, val_seq_precision, val_seq_f1_score]:
            metric.update_state(label, prediction)

    # Update validation metrics
    for metric in [val_video_accuracy, val_video_recall, val_video_precision, val_video_f1_score]:
        metric.update_state(label, tf.nn.softmax(val_accumulated_predictions, axis=-1))

    val_video_loss_metric.update_state(val_sequence_loss_metric.result())

    model.reset_states()

    return


def create_video_dataset(video_path: str, label: tf.Tensor, frame_shape: Tuple[int, int, int] = (240, 320, 3),
                         training=True) -> Tuple[tf.data.Dataset, tf.Tensor]:
    def generator():
        vp = VideoProcessor(video_path, target_shape=frame_shape, training=training)
        for sequence in vp.standardize_fetch_generator(SEQUENCE_LENGTH):
            yield tf.expand_dims(sequence, axis=0)

    video_ds = tf.data.Dataset.from_generator(generator,
                                              output_signature=tf.TensorSpec(shape=(1,
                                                                                    SEQUENCE_LENGTH, frame_shape[0],
                                                                                    frame_shape[1], frame_shape[2]),
                                                                             dtype=tf.float32)
                                              ).map(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

    label_ds = tf.expand_dims(tf.cast(label, tf.float32), axis=0)
    return video_ds, label_ds


def train_model(load_previous_model=False):
    # get global variables
    global train_video_accuracy
    global train_video_recall
    global train_video_precision
    global train_video_f1_score
    global train_seq_accuracy
    global train_seq_recall
    global train_seq_precision
    global train_seq_f1_score
    global val_video_accuracy
    global val_video_recall
    global val_video_precision
    global val_video_f1_score
    global val_seq_accuracy
    global val_seq_recall
    global val_seq_precision
    global val_seq_f1_score
    global train_video_loss_metric
    global val_video_loss_metric
    global train_sequence_loss_metric
    global val_sequence_loss_metric

    global loss_fn
    global optimizer

    # Model parameters
    num_frames, frame_width, frame_height, channels, num_classes = SEQUENCE_LENGTH, 320, 240, 3, 101
    lstm_units = best_hyperparameters['lstm_units']
    dense_units = best_hyperparameters['dense_units']

    if load_previous_model:
        model = load_model(MODEL_SAVE_PATH, custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer,
                                                            'TemporalAttentionLayer': TemporalAttentionLayer,
                                                            'focal_loss_with_probs': focal_loss(gamma=2., alpha=.25, from_logits=False)})
    else:
        model = ActionDetectionModel(1, num_frames, frame_width, frame_height, channels,
                                     num_classes, lstm_units, dense_units,
                                     dropout_rate=DROP_RATE,
                                     fine_tune_until=best_hyperparameters['fine_tune_at'])

    log_dir = "logs/tf_function_training/"

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                        initial_value_threshold=7.81551),
        TensorBoard(log_dir=log_dir + "ActionDetectionModel", histogram_freq=1,
                    update_freq='epoch', write_images=True, embeddings_freq=1),
        EarlyStopping(monitor='val_loss', patience=15, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    for callback in callbacks:
        callback.set_model(model)
        callback.set_params({
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'verbose': 1,
            'do_validation': True,
            'metrics': ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'recall', 'val_recall', 'precision',
                        'val_precision', 'best', 'lr']
        })

    # I am not sure if this is needed. I think it is not needed
    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=['accuracy', 'categorical_accuracy', 'recall', 'precision'])

    logs = {'lr': optimizer.learning_rate.numpy()}
    for callback in callbacks:
        callback.on_train_begin(logs=logs)

    summary_format = ('\rCompleted: {}/{}, ({:.2f}%). Loss: {:.4f}, Accumulative Loss: {:.4f}\tSequence Metrics: {}'
                      '\tVideo Metrics: {}')

    train_sequence_metrics = [train_seq_accuracy, train_seq_recall, train_seq_precision, train_seq_f1_score]
    val_sequence_metrics = [val_seq_accuracy, val_seq_recall, val_seq_precision, val_seq_f1_score]
    train_video_metrics = [train_video_accuracy, train_video_recall, train_video_precision, train_video_f1_score]
    val_video_metrics = [val_video_accuracy, val_video_recall, val_video_precision, val_video_f1_score]

    print('\a')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        for metrics in [train_sequence_metrics, val_sequence_metrics, train_video_metrics, val_video_metrics]:
            for metric in metrics:
                metric.reset_states()

        for loss in [train_video_loss_metric, val_video_loss_metric, train_sequence_loss_metric,
                     val_sequence_loss_metric]:
            loss.reset_states()

        logs = {'epoch': epoch + 1}
        for callback in callbacks:
            callback.on_epoch_begin(epoch, logs=logs)

        train_paths_sample = random.sample(train_paths, len(train_paths) // 10)
        val_path_sample = random.sample(test_paths, len(test_paths) // 10)

        loader = VideoLoader(train_paths_sample, max_queue_size=20, target_shape=(240, 320, 3), training=True)
        loader.start_loading()
        train_portion_size = len(train_paths_sample)
        videos_read = 0
        while videos_read < train_portion_size:
            video_sequence_set, label = loader.get_next_video()
            if video_sequence_set is None:
                break
            train_by_sequence_step(model, video_sequence_set, label)
            videos_read += 1
            sequence_metrics_values = {metric.name: metric.result().numpy() for metric in train_sequence_metrics}
            video_metrics_values = {metric.name: metric.result().numpy() for metric in train_video_metrics}
            sequence_loss_value = train_sequence_loss_metric.result()
            video_loss_value = train_video_loss_metric.result()

            percentage = videos_read / train_portion_size * 100

            output = summary_format.format(videos_read, train_portion_size, percentage, sequence_loss_value,
                                           video_loss_value,
                                           sequence_metrics_values, video_metrics_values)
            print(output, end=' ', flush=True)

        print('\nValidation')
        val_video_read = 0

        val_loader = VideoLoader(val_path_sample, max_queue_size=20, target_shape=(240, 320, 3), training=False)
        val_loader.start_loading()
        val_portion_size = len(val_path_sample)
        while val_video_read < val_portion_size:
            video_sequence_set, label = val_loader.get_next_video()
            if video_sequence_set is None:
                break
            val_by_sequence_step(model, video_sequence_set, label)
            val_video_read += 1
            sequence_metrics_values = {metric.name: metric.result().numpy() for metric in val_sequence_metrics}
            video_metrics_values = {metric.name: metric.result().numpy() for metric in val_video_metrics}
            sequence_loss_value = val_sequence_loss_metric.result()
            video_loss_value = val_video_loss_metric.result()

            percentage = val_video_read / val_portion_size * 100

            output = summary_format.format(val_video_read, val_portion_size, percentage, sequence_loss_value,
                                           video_loss_value,
                                           sequence_metrics_values, video_metrics_values)
            print(output, end=' ', flush=True)

        logs = {}
        logs.update({
            'loss': train_video_loss_metric.result().numpy(),
            'val_loss': val_video_loss_metric.result().numpy(),
            'seq_loss': train_sequence_loss_metric.result().numpy(),
            'val_seq_loss': val_sequence_loss_metric.result().numpy(),
            **{metric.name: metric.result().numpy() for metric in train_sequence_metrics},
            **{metric.name: metric.result().numpy() for metric in val_sequence_metrics},
            **{metric.name: metric.result().numpy() for metric in train_video_metrics},
            **{metric.name: metric.result().numpy() for metric in val_video_metrics},
        })

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

        if model.stop_training:
            break
        print()
    for callback in callbacks:
        callback.on_train_end(logs=logs)

    return model


if __name__ == "__main__":
    train_model(load_previous_model=True)
