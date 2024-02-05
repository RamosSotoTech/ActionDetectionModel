from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import datetime
from dataset.preprocess_datasets.data_iterator import get_dataset_generators, get_train_test_size

DATASET_NUMBERS = [1]
SEQUENCE_LENGTH = 10
BATCH_SIZE = 4
EPOCHS = 30
MODEL_SAVE_PATH = 'model_checkpoint.h5'
train_generator, test_generator = get_dataset_generators(DATASET_NUMBERS, batch_size=4)
_, val_generator = get_dataset_generators([3], batch_size=4)
train_size, test_size = get_train_test_size(DATASET_NUMBERS)


def train_model():
    from models.ADAI import ActionDetectionModel

    # Model parameters
    num_frames, frame_width, frame_height, channels, num_classes = SEQUENCE_LENGTH, 224, 224, 3, 101

    # Initialize your model
    model = ActionDetectionModel(num_frames, frame_width, frame_height, channels, num_classes)

    # Learning Rate Schedule
    initial_learning_rate = 0.01
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

    # Compile the model with learning rate schedule
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Train the model
    model.fit(train_generator,
              steps_per_epoch=train_size // BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=test_generator,
              validation_steps=test_size // BATCH_SIZE,
              callbacks=[checkpoint, tensorboard_callback, early_stopping, reduce_lr])

    return model


def test_model(model):
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(val_generator, verbose=1, steps=test_size // BATCH_SIZE)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_accuracy


if __name__ == "__main__":
    model = train_model()
    test_model(model)
