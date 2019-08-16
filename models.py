from utils import *

from keras.models import Sequential, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, Concatenate, InputLayer
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

from modelmanager import ModelConfig


def _prep_weights(cfg: ModelConfig):
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(cfg._ds.train_y),
                                                np.reshape(cfg._ds.train_y, (cfg._ds.train_y.size,))) \
                if cfg.weighted else np.array([1]*cfg._classes)
    return weights


def bilstm1(cfg : ModelConfig):
    print('simple bilstm')
    rec = partial(recall, w=get_w(cfg._classes))
    prec = partial(precision, w=get_w(cfg._classes))
    metrics = [rec, prec, 'accuracy']

    model = Sequential()
    # model.add(InputLayer(input_shape=(cfg._ds.doc_length, )))
    model.add(Embedding(input_dim=cfg._ds.n_words, output_dim=cfg.embedding_size))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=cfg.lstm_units, return_sequences=True)))
    model.add(Dropout(0.1))
    
    if cfg.with_crf:
        model.add(CRF(cfg._classes))
        model.compile(loss=crf_loss,
                      optimizer=get_optimizer(cfg.optimizer),
                      metrics=metrics)
    else:
        model.add(TimeDistributed(Dense(cfg._classes, activation="softmax")))
        model.compile(loss=weighted_crossentropy(_prep_weights(cfg)),
                      optimizer=get_optimizer(cfg.optimizer),
                      metrics=metrics)
    
    model.summary()
    return model


def bilstm_crf2(cfg : ModelConfig):
    rec = partial(recall, w=get_w(cfg._classes))
    prec = partial(precision, w=get_w(cfg._classes))
    metrics = [rec, prec, 'accuracy']

    inp = Input(shape=(cfg._ds.doc_length, ))

    emb = Embedding(input_dim=cfg._ds.n_words, output_dim=cfg.embedding_size)(inp)

    # lstm
    bilstm = Dropout(0.1)(emb)
    bilstm = Bidirectional(LSTM(units=cfg.lstm_units, return_sequences=True))(bilstm)
    bilstm = Dropout(0.1)(bilstm)
    # td = TimeDistributed(Dense(2, activation="softmax"))(bilstm)

    # concatenate for crf
    # ct = Concatenate()([bilstm, emb])
    crf = CRF(cfg._ds.n_tags, sparse_target=True)(bilstm)

    model = Model(inputs=inp, outputs=crf)
    model.compile(loss=crf_loss,
                  optimizer=get_optimizer(cfg.optimizer),
                  metrics=metrics)

    model.summary()
    return model    

