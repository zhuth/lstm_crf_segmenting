import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.utils import class_weight
from keras.optimizers import Adam

from dataset import DataSet

np.epsilon = lambda: 1e-10

def get_optimizer(name: str):
    if name.startswith('adam:'):
        learning_rate = float(name.split(':')[1])
        return Adam(learning_rate)
    elif name == 'rmsprop':
        return name


def recall(y_true, y_pred, w, k=K):
    true_positives = k.sum(k.round(y_true * y_pred * w))
    possible_positives = k.sum(y_true * w)
    recall = true_positives / possible_positives
    return recall


def precision(y_true, y_pred, w, k=K):
    correctness = k.sum(k.round(y_true * y_pred * w))
    pred_positives = k.sum(k.round(y_pred * w))
    accu = correctness / (pred_positives + k.epsilon())
    return accu


def get_w(n_tags, k=K):
    if k is K:
        w = K.constant
    else:
        w = np.array
    w = w([0] + [1] * (n_tags-1))
    return w


def partial(func, **kwargs):
    def foo(*args):
        return func(*args, **kwargs)
    return foo


def weighted_crossentropy(weights):
    def w_crossentropy(y_true, y_pred):
        return K.mean(tf.nn.weighted_cross_entropy_with_logits(
            y_true,
            y_pred,
            weights,
            name=None
        ), axis=-1)
    return w_crossentropy
