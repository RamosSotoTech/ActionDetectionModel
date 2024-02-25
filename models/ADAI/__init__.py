<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.layers import Multiply, Conv2D
from tensorflow.keras.layers import (TimeDistributed, LSTM, Dense, Dropout, Flatten, GlobalAveragePooling2D,
                                     BatchNormalization, DepthwiseConv2D)
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


class SpatialAttentionLayer(Layer):
    def __init__(self, filters=512, kernel_size=(3, 3), activation='sigmoid', padding='same', **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding

    def build(self, input_shape):
        # Depthwise convolution
        self.depthwise_conv2d = DepthwiseConv2D(kernel_size=self.kernel_size, padding=self.padding,
                                                depth_multiplier=1,  # Controls the output channels per input channel
                                                kernel_initializer='he_normal', use_bias=False)

        # Pointwise convolution to combine the depthwise convolution outputs
        self.pointwise_conv2d = Conv2D(filters=self.filters, kernel_size=(1, 1), padding=self.padding,
                                       kernel_initializer='he_normal', use_bias=False)

        self.batch_norm = BatchNormalization()
        self.activation_layer = Activation(self.activation)

    def call(self, inputs):
        # Apply depthwise convolution
        depthwise = self.depthwise_conv2d(inputs)

        # Apply pointwise convolution
        pointwise = self.pointwise_conv2d(depthwise)

        x = self.batch_norm(pointwise)
        attention = self.activation_layer(x)
        return Multiply()([inputs, attention])

    def compute_output_shape(self, input_shape):
        return input_shape


class TemporalAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], 1),
                                      initializer='glorot_uniform',
                                      trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))

    def call(self, inputs):
        attention_scores = tf.keras.backend.dot(inputs, self.kernel)
        attention_scores = tf.keras.backend.squeeze(attention_scores, -1)
        attention_scores = tf.keras.backend.softmax(attention_scores)
        attended_sequence = inputs * attention_scores[..., None]
        attended_sequence = tf.reduce_sum(attended_sequence, axis=1)
        return attended_sequence


class ResettableLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        super(ResettableLSTM, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        actual_input, reset_indicator = inputs

        # Process 'actual_input' as usual
        output = super().call(actual_input, **kwargs)

        if tf.reduce_any(reset_indicator):
            self.reset_states()

        return output


class SelectiveResetLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs['stateful'] = True  # Make sure the LSTM is stateful
        kwargs['name'] = kwargs.get('name', 'SelectiveResetLSTM')  # Set the name if not provided
        super(SelectiveResetLSTM, self).__init__(*args, **kwargs)
        self.indicators = None  # Indicator tensor to control state reset

    def call(self, inputs, training=None, mask=None, initial_state=None, constants=None):

        # Reset states selectively based on indicators
        if self.indicators is not None:
            # If any indicator in the batch is 1, reset the state
            if tf.reduce_any(tf.cast(self.indicators, dtype=tf.bool)):
                self.reset_states()

        # Proceed with the standard LSTM processing
        return super().call(inputs, training=training, mask=mask, initial_state=initial_state)


# phase 1: train by sequence of frames, stateless LSTM
def ActionDetectionModel_Phase1(num_frames, frame_width, frame_height, channels, num_classes, lstm_units=256,
                         dense_units=1024, dropout_rate=0.5, fine_tune_until=None):
    # Video frame input
    video_input = Input(shape=(num_frames, frame_height, frame_width, channels), name='video_input')

    # Load the VGG19 model
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(frame_height, frame_width, channels))
    if fine_tune_until:
        for layer in base_model.layers[:-fine_tune_until]:
            layer.trainable = False

    # TimeDistributed VGG19 model for frame feature extraction
    td_base_model = TimeDistributed(base_model)(video_input)
    td_attention = TimeDistributed(SpatialAttentionLayer(filters=512, kernel_size=(3, 3), activation='relu'))(
        td_base_model)
    td_pooling = TimeDistributed(GlobalAveragePooling2D())(td_attention)
    td_flatten = TimeDistributed(Flatten())(td_pooling)

    lstm = LSTM(lstm_units, return_sequences=False, stateful=False, dropout=0.5, recurrent_dropout=0.5,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=l1_l2(l2=1e-4), name='lstm')(td_flatten)

    # Following layers
    batch_norm = BatchNormalization()(lstm)
    dense_layer = Dense(dense_units, activation='relu')(batch_norm)
    dropout = Dropout(dropout_rate)(dense_layer)
    predictions = Dense(num_classes, activation='softmax')(dropout)

    # Construct the final model
    model = Model(inputs=video_input, outputs=predictions)

    return model

# todo: Create graph of the model for the documentation
# todo: Add the Masking layer to enable ignoring padding frames, and allow for batch training
# phase 2: train by sequence of videos. Stateful LSTM
def ActionDetectionModel(batch_size, num_frames, frame_width, frame_height, channels, num_classes, lstm_units=256,
                         dense_units=1024, dropout_rate=0.5, fine_tune_until=None):
    # Video frame input
    video_input = Input(batch_shape=(batch_size, num_frames, frame_height, frame_width, channels), name='video_input')

    # Load the VGG19 model
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(frame_height, frame_width, channels))
    if fine_tune_until:
        for layer in base_model.layers[:-fine_tune_until]:
            layer.trainable = False

    # TimeDistributed VGG19 model for frame feature extraction
    td_base_model = TimeDistributed(base_model)(video_input)
    td_attention = TimeDistributed(SpatialAttentionLayer(filters=512, kernel_size=(3, 3), activation='relu'))(
        td_base_model)
    td_pooling = TimeDistributed(GlobalAveragePooling2D())(td_attention)
    td_flatten = TimeDistributed(Flatten())(td_pooling)

    lstm = LSTM(lstm_units, return_sequences=False, stateful=True, dropout=0.5, recurrent_dropout=0.5,
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=l1_l2(l2=1e-4), name='lstm')(td_flatten)

    # Following layers
    batch_norm = BatchNormalization()(lstm)
    dense_layer = Dense(dense_units, activation='relu')(batch_norm)
    dropout = Dropout(dropout_rate)(dense_layer)
    predictions = Dense(num_classes, activation='softmax')(dropout)

    # Construct the final model
    model = Model(inputs=video_input, outputs=predictions)

    return model


# class ActionDetectionModel(Model):
#     def __init__(self, num_frames, frame_width, frame_height, channels, num_classes, lstm_units=256, dense_units=1024, dropout_rate=0.5):
#         super(ActionDetectionModel, self).__init__()
#         self.num_frames = num_frames
#         self.frame_width = frame_width
#         self.frame_height = frame_height
#         self.channels = channels
#         self.num_classes = num_classes
#         self.lstm_units = lstm_units
#         self.dense_units = dense_units
#         self.dropout_rate = dropout_rate
#
#         self.base_model = VGG19(include_top=False, weights='imagenet', input_shape=(frame_height, frame_width, channels))
#         self.td_base_model = TimeDistributed(self.base_model)
#         self.td_attention = TimeDistributed(SpatialAttentionLayer(filters=512, kernel_size=(3, 3), activation='relu'))
#         self.td_pooling = TimeDistributed(GlobalAveragePooling2D())
#         self.td_flatten = TimeDistributed(Flatten())
#         self.lstm = LSTM(lstm_units, return_sequences=False, stateful=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
#         self.batch_norm = BatchNormalization()
#         self.dense = Dense(dense_units, activation='relu')
#         self.dropout = Dropout(dropout_rate)
#         self.final_dense = Dense(num_classes, activation='softmax')
#
#     def call(self, inputs, training=False, mask=None):
#         video_frames, reset_state_indicator = inputs
#         x = self.td_base_model(video_frames)
#         x = self.td_attention(x)
#         x = self.td_pooling(x)
#         x = self.td_flatten(x)
#         x = self.lstm(x)
#         x = self.batch_norm(x)
#         x = self.dense(x)
#         x = self.dropout(x, training=training)
#         output = self.final_dense(x)
#
#         # Reset LSTM states where the indicator is True
#         for i, reset in enumerate(reset_state_indicator):
#             if reset:
#                 self.lstm.reset_states(states=[None for _ in range(len(self.lstm.states))])
#
#         return output
#
#     def reset_states(self):
#         self.lstm.reset_states()
#
#     def train_step(self, data):
#         if len(data) == 3:
#             x, y, sample_weight = data
#         else:
#             x, y = data
#             sample_weight = None
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
#
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update the metrics.
#         # Metrics are configured in `compile()`.
#         self.compiled_metrics.update_state(y, y_pred, sample_weight)
#         # Return a dict mapping metric names to current value.
#         return {m.name: m.result() for m in self.metrics}
#     def test_step(self, data):
#         x_batch_val, y_batch_val = data
#
#         y_pred_val = self(x_batch_val, training=False)
#         loss_val = self.compiled_loss(y_batch_val, y_pred_val)
#         self.compiled_metrics.update_state(y_batch_val, y_pred_val)
#
#         return {m.name: m.result() for m in self.metrics}

=======
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
>>>>>>> parent of 4d41ba2 (Add video handling and preprocessing scripts)
