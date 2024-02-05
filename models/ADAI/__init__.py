from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (TimeDistributed, LSTM, Dense, Dropout, Flatten, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.models import Sequential


def ActionDetectionModel(num_frames, frame_width, frame_height, channels, num_classes, lstm_units=256,
                         dense_units=1024, dropout_rate=0.5, fine_tune_at=None):
    # Load the VGG19 model, excluding the top layers
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(frame_width, frame_height, channels))

    # Freeze the layers of VGG19
    for layer in base_model.layers:
        layer.trainable = False

    # Optional fine-tuning
    if fine_tune_at:
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    # Define the sequential model
    model = Sequential([
        # Add the TimeDistributed layer to process each frame through VGG19
        TimeDistributed(base_model, input_shape=(num_frames, frame_width, frame_height, channels)),
        TimeDistributed(GlobalAveragePooling2D()),  # Reduce dimensions and aggregate features per frame

        # Flatten to prepare for the LSTM
        TimeDistributed(Flatten()),

        # LSTM layer for temporal processing
        LSTM(lstm_units, return_sequences=False),

        # Batch normalization for regularization
        BatchNormalization(),

        # Dense layer for further processing
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),

        # Final prediction layer
        Dense(num_classes, activation='softmax')
    ])

    return model