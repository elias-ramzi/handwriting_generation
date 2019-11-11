import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard

from data import DataSynthesis
from utils import plot_stroke, json_default
from models.handwriting_synthesis import HandWritingSynthesis

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''' CONFIG '''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Run_ID = int(time.time())


MODEL_PATH = 'models/trained/test/model_synthesis_overfit.h5'
EPOCH_MODEL_PATH = 'models/trained/test/model_synthesis_overfit_{}.h5'
LOAD_PREVIOUS = None
DATA_PATH = 'data/strokes-py3.npy'
HISTORY_PATH = 'models/history/test/history_experience.json'

VERBOSE = False

model_kwargs = {
    'regularizer_type': 'l2',
    'reg_mean': 0.,
    'reg_std': 0.,
    'reg_l2': 0.,
    'lr': .0001,
    'rho': .95,
    'momentum': .9,
    'epsilon': .0001,
    'centered': True,
    'inf_type': 'max',
}

HIDDEN_DIM = 400
NUM_LAYERS = 3

data_kwargs = {
    'path_to_data': DATA_PATH,
    'train_split': 0.9
}

train_generator_kwargs = {
    'batch_size': 1,
    'shuffle': False,
}

validation_generator_kwargs = {
    'batch_size': 1,
    'shuffle': True,
}

EPOCHS = 1
STEPS_PER_EPOCH = 1
VAL_STEPS = 1
MODEL_CHECKPOINT = 5

# bias for writing ~~style~~
BIAS = None


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''' TRAIN ''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

D = DataSynthesis(**data_kwargs)
WINDOW_SIZE = len(D.sentences[0][0])

model_kwargs['vocab_size'] = WINDOW_SIZE
hws = HandWritingSynthesis(**model_kwargs)
hws.make_model()
tensorboard_cb = TensorBoard(log_dir='logs/')
tensorboard_cb.set_model(hws.model)

nan = False
generator = D.batch_generator(
    **train_generator_kwargs,
)
validation_generator = D.batch_generator(
    **validation_generator_kwargs,
)

# XXX: use the get_initial_state of WindowedLSTMCell
input_states = [
    # stateh1, statec1
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
    # stateh2, statec2
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
    # stateh3, statec3
    tf.zeros((1, HIDDEN_DIM), dtype=float), tf.zeros((1, HIDDEN_DIM), dtype=float),
    # window kappa
    tf.zeros((1, WINDOW_SIZE), dtype=float), tf.zeros((1, 10), dtype=float),
    # phi, alpha, beta
    tf.zeros((1, 1), dtype=float), tf.zeros((1, 10), dtype=float), tf.zeros((1, 10), dtype=float),
]

history = {
    'train_loss': [],
    'validation_loss': [],
}

try:
    # Test for overfitting
    strokes, sentence, targets = next(generator)
    for e in range(1, EPOCHS + 1):
        train_loss = []
        val_loss = []
        for s in tqdm(range(1, STEPS_PER_EPOCH+1), desc="Epoch {}/{}".format(e, EPOCHS)):
            # strokes, sentence, targets = next(generator)
            loss = hws.train(strokes, sentence, input_states, targets)
            train_loss.append(loss)

            if loss is np.nan:
                nan = True
                print('exiting train @epoch : {}'.format(e))
                break

        for _ in range(VAL_STEPS):
            vstrokes, vsentence, vtargets = next(validation_generator)
            val_loss.append(hws.validation(vstrokes, vsentence, input_states, vtargets))

        mean_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)
        history['train_loss'].append(mean_loss)
        history['validation_loss'].append(mean_val_loss)
        print("Epoch {:03d}: Loss: {:.3f} / Validation loss : {:.3f}"
              .format(e, mean_loss, mean_val_loss))

        if e % MODEL_CHECKPOINT == 0:
            hws.save_weights(EPOCH_MODEL_PATH.format(e))

        if nan:
            break

except KeyboardInterrupt:
    pass

if not nan:
    hws.save_weights(MODEL_PATH)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''''' EVALUATE ''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

verbose_sentence = "".join(D.encoder.inverse_transform(sentence)[0])
strokes1, windows, _, kappas, alphas, betas = hws.infer(
    sentence, inf_type='max',
    verbose=verbose_sentence,
)
phis = hws.windowedlstm.cell.phi
weights = np.stack([np.squeeze(x.numpy()) for x in phis], axis=1)
strokes1[:, 1] = strokes1[:, 1] * D.std1 + D.mean1
strokes1[:, 2] = strokes1[:, 2] * D.std2 + D.mean2
target = tf.gather(targets, [2, 0, 1], axis=2)[0].numpy()
target[:, 1] = target[:, 1] * D.std1 + D.mean1
target[:, 2] = target[:, 2] * D.std2 + D.mean2

with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, default=json_default)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title('Learn curv')
plt.plot(history['train_loss'], label='Training learn curv')
plt.plot(history['validation_loss'], color='r', label='Validation learn curv')
plt.subplot(2, 1, 2)
plt.title('Weights over steps')
plt.imshow(weights, cmap='plasma')
plt.show()

plot_stroke(strokes1)
plot_stroke(target)
import ipdb; ipdb.set_trace()
