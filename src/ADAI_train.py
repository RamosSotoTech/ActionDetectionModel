import cv2
import numpy as np
from keras.src.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K
import datetime
import tensorflow as tf

from tensorflow.keras.callbacks import Callback

from dataset.preprocess_datasets.data_iterator import get_train_test_size
from models.ADAI import ActionDetectionModel, SpatialAttentionLayer, TemporalAttentionLayer
from tensorflow.keras.utils import custom_object_scope



# updated hyperparameters
best_hyperparameters = {'lstm_units': 512, 'dense_units': 1024, 'dropout_rate': 0.5, 'fine_tune_at': 4,
                        'num_frames': 1, 'tuner/epochs': 4, 'tuner/initial_epoch': 2, 'tuner/bracket': 2,
                        'tuner/round': 1, 'tuner/trial_id': '0005'}

SEQUENCE_LENGTH = best_hyperparameters['num_frames']
BATCH_SIZE = 1
EPOCHS = 100
MODEL_SAVE_PATH = '_model_checkpoint.keras'
DROP_RATE = best_hyperparameters['dropout_rate']
TRAINING_DATASET = [1]
VALIDATION_DATASET = [3]

from dataset.preprocess_datasets.testing_generators import get_dataset_generators
from dataset.preprocess_datasets.testing_generators import get_data_from_BatchVideoProcessor

train_generator, test_generator = get_dataset_generators(TRAINING_DATASET, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
_, val_generator = get_dataset_generators(VALIDATION_DATASET, batch_size=BATCH_SIZE)
train_size, test_size = get_train_test_size(TRAINING_DATASET)
_, val_size = get_train_test_size(VALIDATION_DATASET)

# train_generator, test_generator = get_dataset_generators(TRAINING_DATASET, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)


def preprocessing(frame: np.ndarray) -> np.ndarray:
    frame = preprocess_input(cv2.resize(frame, (320, 240)))
    frame = frame / 255.0
    return frame

train_generator, test_generator = get_data_from_BatchVideoProcessor(TRAINING_DATASET, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)


def calculate_video_accuracy(predictions_accumulated, labels_accumulated):
    all_predictions = np.concatenate(predictions_accumulated, axis=0)
    all_labels = np.concatenate(labels_accumulated, axis=0)
    correct_predictions = np.sum(np.argmax(all_predictions, axis=1) == np.argmax(all_labels, axis=1))
    total_items = len(all_labels)
    return correct_predictions / total_items if total_items > 0 else 0

def train_model(load_previous_model=False):

    # Model parameters
    batch_size = BATCH_SIZE
    num_frames, frame_width, frame_height, channels, num_classes = SEQUENCE_LENGTH, 320, 240, 3, 101
    lstm_units = best_hyperparameters['lstm_units']
    dense_units = best_hyperparameters['dense_units']

    if load_previous_model:
        model = load_model(MODEL_SAVE_PATH, custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer,
                                                            'TemporalAttentionLayer': TemporalAttentionLayer})
    else:
        model = ActionDetectionModel(batch_size, num_frames, frame_width, frame_height, channels,
                                     num_classes, lstm_units, dense_units,
                                     dropout_rate=DROP_RATE,
                                     fine_tune_until=best_hyperparameters['fine_tune_at'])

    initial_learning_rate = 0.005
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4, beta_1=0.9, beta_2=0.999,
                                     epsilon=1e-07, amsgrad=False, name='AdamW')

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min', initial_value_threshold=5.0)
    log_dir = "logs/fit_with_AdamW/" + "ActionDetectionModel_attentionMechanism2"

    summary_writer = tf.summary.create_file_writer(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='batch')

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

    # Compile the model with learning rate schedule
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    total_correct = 0
    total_items = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        epoch_loss_avg = tf.keras.metrics.Mean()
        videos_read = 0
        predictions_accumulated = []
        labels_accumulated = []

        # Iterate over the batches of the dataset
        for step, (batch_video_frames, batch_labels, batch_indicators) in enumerate(train_generator):

            if tf.reduce_any(batch_indicators):
                model.get_layer('lstm').reset_states()
                if predictions_accumulated and labels_accumulated:  # If there are accumulated predictions and labels
                    videos_read += 1
                    video_accuracy = calculate_video_accuracy(predictions_accumulated, labels_accumulated)
                    with summary_writer.as_default():
                        tf.summary.scalar('video_accuracy', video_accuracy, step=videos_read)
                    print(f'Video {videos_read}: Accuracy: {video_accuracy:.4f}')

                predictions_accumulated = []
                labels_accumulated = []

            with tf.GradientTape() as tape:
                predictions = model(batch_video_frames, training=True)
                loss_value = loss_fn(batch_labels, predictions)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)

            correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_labels, axis=1))
            total_correct += correct_predictions
            total_items += len(batch_labels)

            accuracy = total_correct / total_items

            predictions_accumulated.append(predictions)
            labels_accumulated.append(batch_labels)

            with summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss_avg.result(), step=step)
                tf.summary.scalar('metrics', videos_read / train_size, step=step)
                tf.summary.scalar('accuracy', accuracy, step=step)

            if step % 20 == 0:
                print(f'Step {step}: Loss: {epoch_loss_avg.result().numpy()}')
                print(f'Percents: {videos_read / train_size * 100:.2f}%')

        if predictions_accumulated and labels_accumulated:
            video_accuracy = calculate_video_accuracy(predictions_accumulated, labels_accumulated)
            with summary_writer.as_default():
                tf.summary.scalar('video_accuracy', video_accuracy, step=videos_read)
            print(f'Video {videos_read}: Accuracy: {video_accuracy:.4f}')

        print(f'Epoch Loss: {epoch_loss_avg.result().numpy()}')

        # validation loop
        val_loss_avg = tf.keras.metrics.Mean()
        val_total_correct = 0
        val_total_items = 0
        val_videos_read = 0
        val_predictions_accumulated = []
        val_labels_accumulated = []

        for val_step, (val_batch_video_frames, val_batch_labels, val_batch_indicators) in enumerate(val_generator):
            if tf.reduce_any(val_batch_indicators):
                model.get_layer('lstm').reset_states()
                if val_predictions_accumulated and val_labels_accumulated:
                    val_video_accuracy = calculate_video_accuracy(val_predictions_accumulated, val_labels_accumulated)
                    with summary_writer.as_default():
                        tf.summary.scalar('val_video_accuracy', val_video_accuracy, step=val_step)
                    print(f'Val Video {val_videos_read}: Accuracy: {val_video_accuracy:.4f}')
                    val_videos_read += 1

                val_predictions_accumulated = []
                val_labels_accumulated = []

            val_predictions = model(val_batch_video_frames, training=False)
            val_loss_value = loss_fn(val_batch_labels, val_predictions)
            val_loss_avg.update_state(val_loss_value)

            # Calculate accuracy
            val_correct_predictions = np.sum(np.argmax(val_predictions, axis=1) == np.argmax(val_batch_labels, axis=1))
            val_total_correct += val_correct_predictions
            val_total_items += len(val_batch_labels)

            val_accuracy = val_total_correct / val_total_items

            val_predictions_accumulated.append(val_predictions)
            val_labels_accumulated.append(val_batch_labels)

            # Log loss and metric to TensorBoard during validation
            with summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss_avg.result(), step=val_step)
                tf.summary.scalar('val_accuracy', val_accuracy, step=val_step)

        print(f'Validation Loss: {val_loss_avg.result().numpy()}')
        monitor_value = val_loss_avg.result().numpy()
        if checkpoint.monitor_op(checkpoint.best, monitor_value):
            # If it improved, save the model to disk
            checkpoint.best = monitor_value
            model.save(MODEL_SAVE_PATH, overwrite=True)

        # execute Callbacks
        checkpoint.on_epoch_end(epoch, logs={"val_loss": val_loss_avg.result().numpy()})
        tensorboard_callback.on_epoch_end(epoch, logs={"loss": epoch_loss_avg.result().numpy(),
                                                       "val_loss": val_loss_avg.result().numpy()})
        early_stopping.on_epoch_end(epoch, logs={"val_loss": val_loss_avg.result().numpy()})
        reduce_lr.on_epoch_end(epoch, logs={"val_loss": val_loss_avg.result().numpy()})

        # Check for early stopping
        if early_stopping.model.stop_training:
            break

    return model

if __name__ == "__main__":
    train_model(load_previous_model=False)

# def train_model(model, train_generator, val_generator, epochs, optimizer, loss_fn, callbacks=None):
#     # Compile the model with the optimizer and loss function
#     model.compile(optimizer=optimizer, loss=loss_fn)
#
#     for epoch in range(epochs):
#         print(f'Epoch {epoch + 1}/{epochs}')
#         epoch_loss_avg = tf.keras.metrics.Mean()
#         epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
#
#         # Training loop
#         for step, (batch_video_frames, batch_labels, batch_indicators) in enumerate(train_generator):
#             # # Reset LSTM states for new videos
#             # if tf.reduce_any(batch_indicators):
#             #     model.reset_states()  # Ensure your model has a reset_states method
#
#             with tf.GradientTape() as tape:
#                 predictions = model([batch_video_frames, batch_indicators], training=True)
#                 loss_value = loss_fn(batch_labels, predictions)
#
#             grads = tape.gradient(loss_value, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#             # Update training metrics
#             epoch_loss_avg.update_state(loss_value)
#             epoch_accuracy.update_state(batch_labels, predictions)
#
#             if step % 100 == 0:
#                 print(
#                     f'Step {step}: Loss: {epoch_loss_avg.result().numpy()}, Accuracy: {epoch_accuracy.result().numpy()}')
#
#         # End of epoch - Log epoch metrics
#         print(
#             f'Epoch {epoch + 1}: Loss: {epoch_loss_avg.result().numpy()}, Accuracy: {epoch_accuracy.result().numpy()}')
#
#         # Validation loop
#         val_loss_avg = tf.keras.metrics.Mean()
#         val_accuracy = tf.keras.metrics.CategoricalAccuracy()
#
#         for batch_video_frames, batch_labels, batch_indicators in val_generator:
#             # Reset LSTM states for new videos
#             if tf.reduce_any(batch_indicators):
#                 model.reset_states()
#
#             val_predictions = model(batch_video_frames, training=False)
#             val_loss_value = loss_fn(batch_labels, val_predictions)
#
#             # Update validation metrics
#             val_loss_avg.update_state(val_loss_value)
#             val_accuracy.update_state(batch_labels, val_predictions)
#
#         # Log validation metrics
#         print(f'Validation Loss: {val_loss_avg.result().numpy()}, Validation Accuracy: {val_accuracy.result().numpy()}')
#
#         # Execute callbacks at the end of an epoch, if any
#         if callbacks:
#             for callback in callbacks:
#                 callback.on_epoch_end(epoch, logs={'loss': epoch_loss_avg.result().numpy(),
#                                                    'accuracy': epoch_accuracy.result().numpy(),
#                                                    'val_loss': val_loss_avg.result().numpy(),
#                                                    'val_accuracy': val_accuracy.result().numpy()})
#
#     return model
#
#
# if __name__ == "__main__":
#     load_previous_model = False
#     batch_size = BATCH_SIZE
#     num_frames, frame_width, frame_height, channels, num_classes = SEQUENCE_LENGTH, 320, 240, 3, 101
#     lstm_units = best_hyperparameters['lstm_units']
#     dense_units = best_hyperparameters['dense_units']
#     model = ActionDetectionModel(num_frames, frame_width, frame_height, channels,
#                                  num_classes, lstm_units, dense_units, dropout_rate=DROP_RATE)
#     initial_learning_rate = 0.005
#     lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
#     optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4, beta_1=0.9, beta_2=0.999,
#                                      epsilon=1e-07, amsgrad=False, name='AdamW')
#     loss_fn = tf.keras.losses.CategoricalCrossentropy()
#     checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#     early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit_with_AdamW/" + "ActionDetectionModel_attentionMechanism2",
#                                                           histogram_freq=1, write_graph=True, update_freq='batch')
#     callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard_callback]
#
#     model.compile(optimizer=optimizer, loss=loss_fn)
#     model = train_model(model, train_generator, val_generator, EPOCHS, optimizer, loss_fn, callbacks=callbacks)
