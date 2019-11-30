"""
Implementing the Handwriting Prediction section of
the paper found here https://arxiv.org/pdf/1308.0850.pdf by Alex Graves
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM,
    Dense, Concatenate
)
from tqdm import tqdm

from models.base_model import BaseModel


class HandWritingPrediction(BaseModel):

    def __init__(
        self,
        lstm='stacked',
        lr=0.0001,
        rho=0.95,
        momentum=0.9,
        epsilon=0.0001,
        centered=True,
        verbose=True,
    ):
        assert lstm in ['single', 'stacked'],\
            "You should pass either lstm='single or stacked'"
        self.lstm = lstm
        self.verbose = verbose

        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

        self.optimizer = tf.keras.optimizers.RMSprop(
                lr=self.lr,
                rho=self.rho,
                momentum=self.momentum,
                epsilon=self.epsilon,
                centered=self.centered,
                clipvalue=10,
        )

    def _make_model(self):
        strokes = Input((None, 3))
        output_states = []

        if self.lstm == 'single':
            stateh1 = Input(900)
            statec1 = Input(900)
            input_states = [stateh1, statec1]
            self.num_layers = 1
            self.hidden_dim = 900
            lstm, stateh1, statec1 = LSTM(
                900,
                name='h1',
                return_sequences=True,
                return_state=True,
            )(strokes, initial_state=input_states)
            output_states += [stateh1, statec1]

        elif self.lstm == 'stacked':
            stateh1 = Input(400)
            statec1 = Input(400)
            stateh2 = Input(400)
            statec2 = Input(400)
            stateh3 = Input(400)
            statec3 = Input(400)
            input_states = [stateh1, statec1, stateh2, statec2, stateh3, statec3]
            self.num_layers = 3
            self.hidden_dim = 400
            lstm1, ostateh1, ostatec1 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                name='h1',
                )(strokes, initial_state=input_states[0:2])
            output_states += [ostateh1, ostatec1]

            # skip1 = Dense(400, name='Wih2', use_bias=False)(strokes)
            _input2 = Concatenate(name='Skip1')([strokes, lstm1])
            lstm2, ostateh2, ostatec2 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                name='h2',
            )(_input2, initial_state=input_states[2:4])
            output_states += [ostateh2, ostatec2]

            # skip2 = Dense(400, name='Wih3', use_bias=False)(strokes)
            _input3 = Concatenate(name='Skip2')([strokes, lstm2])
            lstm3, ostateh3, ostatec3 = LSTM(
                400,
                return_sequences=True,
                return_state=True,
                name='h3',
                )(_input3, initial_state=input_states[4:6])
            output_states += [ostateh3, ostatec3]

            # skip31 = Dense(400, name='Wh1y', use_bias=False)(lstm1)
            # skip32 = Dense(400, name='Wh2y', use_bias=False)(lstm2)
            lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm3])

        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(
            inputs=[strokes, input_states],
            outputs=[mixture_coefs, output_states]
        )

        return model

    def make_model(self, load_weights=None):
        self.model = self._make_model()
        if load_weights is not None:
            self.model.load_weights(load_weights)
        if self.verbose:
            self.model.summary()

    def train(self, inputs, targets, load_weights=None):
        """
        adapted from
        https://github.com/tensorflow/tensorflow/issues/28707
        """
        if not hasattr(self, 'model'):
            self.make_model(load_weights=load_weights)

        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            predictions = outputs[0]
            targets = tf.dtypes.cast(targets, dtype=float)
            loss = self.loss_function(targets, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)

        # # Clips gradient for output Dense layer
        # gradients[-1] = tf.clip_by_value(gradients[-1], -100.0, 100.0)
        # gradients[-2] = tf.clip_by_value(gradients[-2], -100.0, 100.0)
        #
        # # Clips gradient for LSTM layers
        # for i, grad in enumerate(gradients[:-2]):
        #     gradients[i] = tf.clip_by_value(gradients[i], -10.0, 10.0)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def infer(self, seed=None, weights_path=None, reload=False):
        if not hasattr(self, 'model') or reload:
            self.make_model(load_weights=weights_path)
        np.random.seed(seed)
        length = np.random.randint(400, 1200)
        print()
        print("Generating a random sentence of \033[92m {}\033[00m strokes".format(length))
        print()

        X = tf.zeros((1, 1, 3))
        states = [tf.zeros((1, self.hidden_dim))] * 2 * self.num_layers
        strokes = []
        for _ in tqdm(range(length), desc='Creating a series of strokes'):
            mixture_coefs, states = self.model([X, states], training=False)
            end_stroke, x, y = self._infer(mixture_coefs)
            X = np.array([x, y, end_stroke]).reshape((1, 1, 3))
            strokes.append((end_stroke, x, y))
        return np.vstack(strokes)
