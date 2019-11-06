import numpy

from data import DataSynthesis
from models import HandWritingPrediction, HandWritingSynthesis
strokes = numpy.load('../data/strokes-py3.npy', allow_pickle=True)
stroke = strokes[0]

HWP = HandWritingPrediction()
HWP.make_infer_model(weights_path='../models/trained/generation_network.h5')

D = DataSynthesis(path_to_sentences='../data/sentences.txt')
_ = D.sentences
HWS = HandWritingSynthesis()
HWS.make_model(load_weights='../models/trained/synthesis_network.h5')


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
    sentence = D.prepare_text(text)
    stroke, _, _, _ = HWS.infer(sentence, seed=random_seed)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
