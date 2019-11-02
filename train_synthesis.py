import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data import DataPrediction
from utils import plot_stroke
from models import HandWritingSynthesis

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''CONFIG'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_PATH = 'models/trained/model_synthesis_overfit.h5'
EPOCH_MODEL_PATH = 'models/trained/model_synthesis_overfit_{epoch}.h5'
LOAD_PREVIOUS = None
DATA_PATH = 'data/strokes-py3.npy'

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

HIDDEN_DIM = 900 if model_kwargs['lstm'] == 'single' else 400
NUM_LAYERS = 1 if model_kwargs['lstm'] == 'single' else 3

data_kwargs = {
    'path_to_data': DATA_PATH,
}

train_generator_kwargs = {
    'batch_size': 1,
    'shuffle': True,
}

EPOCHS = 150
STEPS_PER_EPOCH = 200

# bias for writing ~~style~~
BIAS = None


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''TRAIN'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

D = DataPrediction(**data_kwargs)
hws = HandWritingSynthesis(D.sentences.shape[1], **model_kwargs)
hws.make_model(load_weights=LOAD_PREVIOUS)

nan = False
generator = D.batch_generator(
    **train_generator_kwargs,
)

input_states = [tf.zeros((train_generator_kwargs['batch_size'], D.sentences.shape[1]))]
input_state = tf.zeros((train_generator_kwargs['batch_size'], HIDDEN_DIM))
input_states += [input_state] * 2 * NUM_LAYERS
try:
    # Test for overfitting
    # strokes, targets = next(generator)
    for e in range(1, EPOCHS + 1):
        train_loss = []
        for s in tqdm(range(1, STEPS_PER_EPOCH+1), desc=f"Epoch {e}/{EPOCHS}"):
            strokes, sentence, targets = next(generator)
            loss = hws.train(strokes, input_states, targets)
            train_loss.append(loss)

            if loss is np.nan:
                nan = True
                print(f'exiting train @epoch : {e}')
                break

        mean_loss = np.mean(train_loss)
        print(f"Epoch {e:03d}: Loss: {mean_loss:.3f}")

        if e % 1 == 0:
            hws.model.save_weights(f'models/trained/model_overfit_{e}.h5')

        if nan:
            break

except KeyboardInterrupt:
    pass

if not nan:
    hws.model.save_weights(MODEL_PATH)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''''''EVALUATE'''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

strokes1 = hws.infer(700)
plot_stroke(strokes1)
strokes2 = hws.infer(700, 'sum')
plot_stroke(strokes2)
import ipdb; ipdb.set_trace()
