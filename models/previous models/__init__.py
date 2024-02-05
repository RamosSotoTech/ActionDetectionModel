import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Flatten, Bidirectional, Layer
from tensorflow.keras.regularizers import l2


class LSTMPeepholes(LSTM):
    def __init__(self, units, **kwargs):
        super(LSTMPeepholes, self).__init__(units, **kwargs)

    def build(self, input_shape):
        super(LSTMPeepholes, self).build(input_shape)
        self.W_o = self.add_weight((self.units, self.units),
                                   name='W_o',
                                   initializer=self.recurrent_initializer)
        self.W_f = self.add_weight((self.units, self.units),
                                   name='W_f',
                                   initializer=self.recurrent_initializer)
        self.W_i = self.add_weight((self.units, self.units),
                                   name='W_i',
                                   initializer=self.recurrent_initializer)

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        z = K.dot(inputs, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z1 + K.dot(c_tm1, self.W_i))  # input gate with peephole
        f = self.recurrent_activation(z2 + K.dot(c_tm1, self.W_f))  # forget gate with peephole
        o = self.recurrent_activation(z3 + K.dot(c_tm1, self.W_o))  # output gate with peephole

        c = f * c_tm1 + i * self.activation(z0)
        h = o * self.activation(c)
        return h, [h, c]

    # Outbound nodes need to be updated in TensorFlow version 2.0
    def __setattr__(self, name, value):
        if 'outbound_nodes' in self.__dict__ and name == 'outbound_nodes':
            pass
        else:
            Layer.__setattr__(self, name, value)


class ActionDetectionModel(tf.keras.Model):
    def __init__(self, num_frames, frame_width, frame_height, channels, num_classes,
                 lstm_units=256, dense_units=1024, dropout_rate=0.5, dense_activation='relu', **kwargs):
        super(ActionDetectionModel, self).__init__(**kwargs)
        self.num_frames = num_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.channels = channels
        self.num_classes = num_classes

        pre_trained_model = VGG19(include_top=False, weights='imagenet',
                                  input_shape=(self.frame_width, self.frame_height, self.channels))
        for layer in pre_trained_model.layers[:-4]:
            layer.trainable = False

        self.time_distributed = TimeDistributed(pre_trained_model,
                                                input_shape=(
                                                self.num_frames, self.frame_width, self.frame_height, self.channels))
        self.flatten = TimeDistributed(Flatten())
        self.lstm1 = Bidirectional(
            LSTMPeepholes(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))
        self.lstm2 = Bidirectional(LSTMPeepholes(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
        self.dense1 = Dense(dense_units, activation=dense_activation, kernel_regularizer=l2(0.01))
        self.dropout = Dropout(dropout_rate)
        self.dense2 = Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.time_distributed(inputs)
        x = self.flatten(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)
