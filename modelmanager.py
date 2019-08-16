import pickle
import yaml
import os
import glob

from keras.utils import to_categorical
from utils import *

class ModelConfig:

    _model_provider = None

    def __init__(self, name='default'):
        self.name = name
        self.dataset = 'data.pkl'
        self.binary_class = False
        
        self.train_batch_size = 50
        self.test_batch_size = 500
        
        self.lstm_units = 256
        self.time_step = 32
        self.embedding_size = 128
        self.with_crf = False

        self.optimizer = 'rmsprop'
        
        self.epoch_num = 20
        self.weighted = False
        self.migrate = {}

    def load(self, cfg=None):
        if cfg is None:
            cfg = '{}.yaml'.format(self.name)
        if isinstance(cfg, str):
            with open(cfg) as fi:
                y = yaml.safe_load(fi)
            y['name'] = os.path.basename(cfg).rsplit('.')[0]
            cfg = y
        for k, v in cfg.items():
            if hasattr(self, k): setattr(self, k, v)
    
    def save_config(self):
        with open('{}.yaml'.format(self.name), 'w') as fo:
            yaml.dump({k: v for k, v in self.__dict__.items() if not k.startswith('_') and isinstance(v, (dict, int, bool, str))}, fo)
        
    def __enter__(self, *args):
        self.model = self.get_model()

        if not self.resume_from_checkpoint():
            if self.migrate:
                if isinstance(self.migrate, str): self.migrate = { self.migrate : {} }

                for conf, layers in self.migrate.items():
                    if not layers:
                        pkls = glob.glob(self._dumped_layer_pkl(conf, '*'))
                        for p in pkls:
                            layer = os.path.basename(p)[:-len('.layer.pkl')]
                            layers[layer] = layer
                    
                    for layer, other_layer_name in layers.items():
                        self.model.get_layer(layer).set_weights(self.load_layer_weights(conf, other_layer_name))

        return self.model

    def __exit__(self, *args, **kwargs):
        pass

    def checkpoints(self):
        return glob.glob('%s/weights-*' % self.name)

    def resume_from_checkpoint(self, ckpt=''):
        if ckpt: 
            ckpts = [ckpt]
        else: 
            ckpts = self.checkpoints()
        
        if ckpts:
            ckpt = max([(os.path.getmtime(_), _) for _ in ckpts])[1]
            print('loading from latest weights', ckpt)
            self.model.load_weights(ckpt)
            return ckpt

    def _dumped_layer_pkl(self, config_name, layer_name):
        return f'{config_name}/{layer_name}.layer.pkl'

    def dump_layer_weights(self):        
        for l in model.layers:
            with open(self._dumped_layer_pkl(self.name, l.name), 'wb') as fo:
                pickle.dump(l.get_weights(), fo)

    def load_layer_weights(self, config_name, layer_name):
        f = self._dumped_layer_pkl(config_name, layer_name)
        if not os.path.exists(f):
            return
        with open(f, 'rb') as fi:
            return pickle.load(fi)

    def get_model(self):
        if self._model_provider is None:
            raise ValueError('Please specify model provider first.')

        self._ds = DataSet(self.dataset)
        self._classes = self._ds.n_tags

        if self.binary_class:
            self._ds.train_y, self._ds.test_y = self._ds.train_y > 0, self._ds.test_y > 0
            self._classes = 2

        return ModelConfig._model_provider(self)

    def train(self):
        from keras.callbacks import ModelCheckpoint, TensorBoard
        checkpoint = [
            ModelCheckpoint("%s/weights-%s-{epoch:02d}-{val_loss:.2f}.hdf5" % (self.name, self.name), 
            monitor='val_rec', verbose=1, save_best_only=False),
            TensorBoard(log_dir="%s/" % self.name, write_graph=False)
            ]
        try:
            return self.model.fit(self._ds.train_x, to_categorical(self._ds.train_y, self._classes), 
                    batch_size=self.train_batch_size, epochs=self.epoch_num,
                    # class_weight=,
                    validation_split=0.1, verbose=1, callbacks=checkpoint)
        except KeyboardInterrupt:
            self.model.save("%s/weights-interrupted.hdf5" % self.name)

    def evaluate(self):
        return self.model.evaluate(self._ds.test_x,
            to_categorical(self._ds.test_y, self._classes), batch_size=self.test_batch_size)

    def test(self, count):
        w = get_w(self._classes, np)
        for _c in range(count):
            i = np.random.randint(len(self._ds.test_x))
            test_x = self._ds.test_x[i]
            test_y = self._ds.test_y[i]
            test_y_c = to_categorical(test_y)
            p = self.model.predict(np.array([test_x]))
            p0 = np.argmax(p[0], axis=-1)
            print('>>', self._ds.visualize(test_x, p0))
            print('==', self._ds.visualize(test_x, test_y))
            print('  P', precision(test_y_c, p, w, np), 
                    'R', recall(test_y_c, p, w, np))
