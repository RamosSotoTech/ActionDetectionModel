from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import TimeDistributed, Flatten, LSTM, BatchNormalization, Dense, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from kerastuner.tuners import Hyperband
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from utils.dir_utils import model_dir
from dataset.preprocess_datasets.data_iterator import get_dataset_generators, get_train_test_size
import os
from datetime import datetime

# Load the dataset
BATCH_SIZE = 4
train_generator, val_generator = get_dataset_generators([3], batch_size=BATCH_SIZE)
train_size, val_size = get_train_test_size([3])


def build_model(hp):
    # Hyperparameters to tune
    lstm_units = hp.Int('lstm_units', min_value=128, max_value=512, step=64)
    dense_units = hp.Int('dense_units', min_value=512, max_value=2048, step=512)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
    fine_tune_until = hp.Choice('fine_tune_at', values=[-1, 4, 50, 150, 200])
    num_frames = hp.Choice('num_frames', values=[10, 20, 30, 60])
    frame_width, frame_height, channels = 224, 224, 3
    num_classes = 101

    # Model definition (similar to your provided code, with the hyperparameters)
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(frame_width, frame_height, channels))

    # Freeze the layers of VGG19
    for layer in base_model.layers:
        layer.trainable = False

    # Optional fine-tuning
    if 0 < fine_tune_until < base_model.layers.__len__():
        for layer in base_model.layers[:fine_tune_until]:
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

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_best_hyperparameters(tuner, train_generator, val_generator):
    # Search for the best hyperparameters
    tuner.search(train_generator, validation_data=val_generator, epochs=10)

    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hyperparameters


def get_best_model(tuner, train_generator, val_generator):
    # Search for the best hyperparameters
    tuner.search(train_generator, validation_data=val_generator, epochs=10)

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


def optimize_hyperparameters():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "hyperparam_tuning", current_time)

    # get model weights directory
    hp_model_weight_dir = os.path.join(model_dir, 'model_weights')
    checkpoint_dir = os.path.join(hp_model_weight_dir, 'HPmodel_checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Callbacks
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        directory=log_dir,  # Use the same log directory for organization
        project_name='action_detection'
    )

    # Todo: Use K-fold cross-validation to avoid overfitting
    # Start the hyperparameter search process
    tuner.search(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=train_size // BATCH_SIZE,
        validation_steps=val_size // BATCH_SIZE,
        epochs=10,
        callbacks=[checkpoint, tensorboard_callback, early_stopping, reduce_lr]
    )

    # Retrieve the best hyperparameters and model
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    return best_hyperparameters, best_model



if __name__ == "__main__":
    best_hyperparameters, best_model = optimize_hyperparameters()
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best model: {best_model.summary()}")
    # save the best model
    best_model.save(os.path.join(model_dir, 'best_model.keras'))

    # save the best hyperparameters in a file
    with open('best_hyperparameters.txt', 'w') as file:
        file.write(str(best_hyperparameters.get_config()))
