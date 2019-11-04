import numpy

from models import HandWritingPrediction
strokes = numpy.load('../data/strokes-py3.npy', allow_pickle=True)
stroke = strokes[0]

HWP = HandWritingPrediction()
HWP.make_infer_model(weights_path='../models/trained/stacked_600_epochs.h5')


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    stroke = HWP.infer(seed=random_seed, inf_type='max')
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
