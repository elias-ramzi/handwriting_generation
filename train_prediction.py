import os
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm

from data import DataPrediction
from utils import plot_stroke, json_default
from models import HandWritingPrediction

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''CONFIG'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_ID = int(time.time())

LOAD_PREVIOUS = None

MODEL_PATH = 'models/trained/test/model_generation_{}.h5'.format(RUN_ID)
EPOCH_MODEL_PATH = 'models/trained/test/model_generation_{}_{}.h5'.format(RUN_ID, "{}")
HISTORY_PATH = 'models/history/test/history_generation_{}.json'.format(RUN_ID)
LOG_PATH = 'models/logs/'

DATA_PATH = 'data/strokes-py3.npy'

VERBOSE = False

model_kwargs = {
    'lstm': 'stacked',
    'lr': .0001,
    'rho': .95,
    'momentum': .9,
    'epsilon': .0001,
    'centered': True,
    'verbose': VERBOSE,
}

HIDDEN_DIM = 900 if model_kwargs['lstm'] == 'single' else 400
NUM_LAYERS = 1 if model_kwargs['lstm'] == 'single' else 3

data_kwargs = {
    'path_to_data': DATA_PATH,
    'train_split': 0.9,
    'scale': True,
}

train_generator_kwargs = {
    'shuffle': False,
}

validation_generator_kwargs = {
    'shuffle': True,
}

EPOCHS = 100
STEPS_PER_EPOCH = 10
VAL_STEPS = 0
MODEL_CHECKPOINT = 1


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''TRAIN'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

D = DataPrediction(**data_kwargs)

hwp = HandWritingPrediction(**model_kwargs)
hwp.make_model(load_weights=LOAD_PREVIOUS)
tensorboard_cb = TensorBoard(log_dir=LOG_PATH)
tensorboard_cb.set_model(hwp.model)

nan = False
generator = D.batch_generator(
    **train_generator_kwargs,
)
validation_generator = D.batch_generator(
    **validation_generator_kwargs,
)

input_state = tf.zeros((1, HIDDEN_DIM))
input_states = [input_state] * 2 * NUM_LAYERS


history = {
    'train_loss': [],
    'validation_loss': [],
}

print()
print("Running HandWritingPrediction train with ID \033[92m {}\033[00m".format(RUN_ID))
print()


try:
    # Test for overfitting
    strokes, targets = next(generator)
    for e in range(1, EPOCHS + 1):
        train_loss = []
        val_loss = []
        for s in tqdm(range(1, STEPS_PER_EPOCH+1), desc="Epoch {}/{}".format(e, EPOCHS)):
            # strokes, sentence, targets = next(generator)
            loss = hwp.train([strokes, input_states], targets)
            train_loss.append(loss)

            if loss is np.nan:
                nan = True
                print('exiting train @epoch : {}'.format(e))
                break

        for _ in range(VAL_STEPS):
            vstrokes, vtargets = next(validation_generator)
            val_loss.append(hwp.validation([vstrokes, input_states], vtargets))

        mean_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)
        history['train_loss'].append(mean_loss)
        history['validation_loss'].append(mean_val_loss)
        print("Epoch {:03d}: Loss:\033[93m {:.3f}\033[00m / Validation loss : {:.3f}"
              .format(e, mean_loss, mean_val_loss))

        if e % MODEL_CHECKPOINT == 0:
            hwp.model.save_weights(EPOCH_MODEL_PATH.format(e))

        if nan:
            break

except KeyboardInterrupt:
    pass

if not nan:
    hwp.model.save_weights(MODEL_PATH)


with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, default=json_default)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''''''EVALUATE'''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
strokes1 = hwp.infer(seed=23)
strokes1 = D.scale_back(strokes1)

plt.figure(figsize=(10, 5))
plt.title('Learning curv')
plt.plot(history['train_loss'], label='Training learn curv')
plt.plot(history['validation_loss'], color='r', label='Validation learn curv')
plt.show()

plot_stroke(strokes1)

import ipdb; ipdb.set_trace()
