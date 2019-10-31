import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TerminateOnNaN,
    TensorBoard,
)

from data import DataPrediction
from utils import plot_stroke
from models.handwriting_prediction import HandWritingPrediction

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''CONFIG'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

MODEL_PATH = 'models/trained/model_stacked_overfit.h5'
EPOCH_MODEL_PATH = 'models/trained/model_overfit_{epoch}.h5'
LOAD_PREVIOUS = 'models/trained/model_stacked_overfit.h5'
SEQUENCE_LENGTH = 700

DATA_PATH = 'data/strokes-py3.npy'

VERBOSE = False

model_kwargs = {
    'train_seq_length': SEQUENCE_LENGTH,
    '_compile': True,
    'lstm': 'single',
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

data_kwargs = {
    'path_to_data': DATA_PATH,
}

train_generator_kwargs = {
    'sequence_lenght': SEQUENCE_LENGTH,
    'batch_size': 1,
}

callbacks = [
    ModelCheckpoint(
        EPOCH_MODEL_PATH,
        save_weights_only=True,
        period=5,
    ),
    TerminateOnNaN(),
]
if VERBOSE:
    callbacks.append(
        TensorBoard(
            'models/logs',
            histogram_freq=5,
        )
    )
fit_kwargs = {
    'steps_per_epoch': 100,
    'epochs': 10,
    'callbacks': callbacks,
}


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''TRAIN'''''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

D = DataPrediction(**data_kwargs)
hwp = HandWritingPrediction(**model_kwargs)
hwp.make_model(load_weights=LOAD_PREVIOUS)

nan = False
try:
    if model_kwargs['_compile']:

        hwp.model.fit_generator(
            generator=D.batch_generator(
                **train_generator_kwargs,
            ),
            **fit_kwargs,
        )

    else:
        generator = D.batch_generator(
                **train_generator_kwargs,
            )
        for e in range(1, fit_kwargs['epochs'] + 1):
            train_loss = []
            for s in tqdm(range(fit_kwargs['steps_per_epoch']), desc=f"Epoch {e}/{fit_kwargs['epochs']}"):
                strokes, targets = next(generator)
                loss = hwp.train(strokes, targets)
                train_loss.append(loss)

                if loss is np.nan:
                    nan = True
                    print(f'exiting train @epoch : {e}')
                    break

            mean_loss = np.mean(train_loss)
            print(f"Epoch {e:03d}: Loss: {mean_loss:.3f}")

            if e % 5 == 0:
                hwp.model.save_weights(f'models/trained/model_overfit_{e}.h5')

            if nan:
                break

except KeyboardInterrupt:
    pass

if not nan:
    hwp.model.save_weights(MODEL_PATH)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# '''''''''''''''''''''''''''''''EVALUATE'''''''''''''''''''''''''''''''
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

strokes1 = hwp.infer(700)
# strokes2 = hwp.infer(700, 'sum')
plot_stroke(strokes1)
# plot_stroke(strokes2)
import ipdb; ipdb.set_trace()
