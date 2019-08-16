import re
import os
import math
import glob
import argparse
import numpy as np
import tensorflow as tf

from dataset import *
from modelmanager import ModelConfig
from models import bilstm1, bilstm_crf2

BIAODIAN = [''] + list('，。、？：；')
ModelConfig._model_provider = bilstm1

def main(cfg, opts):    
    if opts.prepare != 0:
        DataPreparer(BIAODIAN, config.max_length, config.dataset).prepare(limit=opts.prepare)
        exit()

    if cfg.with_crf and not cfg.binary_class:
        ModelConfig._model_provider = bilstm_crf2
        
    with cfg as model:
        if opts.dump:
            cfg.dump()
            exit()
    
        if opts.test:
            cfg.test(10)
            evals = cfg.evaluate()
            print(cfg.name)
            print('_'*32)
            print('loss\trecall\tprec\taccu\n' + \
                '\t'.join([str(_) for _ in evals])
                )
            print('F-score', 2.0/(1.0/evals[1]+1.0/evals[2]))
            exit()

        cfg.train()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BiLSTM CRF trainer/tester')
    parser.add_argument('--ckpt', help='checkpoint dir', default='', type=str)
    parser.add_argument('--test', help='test', dest='test', action='store_true')
    parser.add_argument('--prepare', help='prepare', default=0, type=int)
    parser.add_argument('--config', help='config file', type=str)
    parser.add_argument('--cpu', help='cpu only', dest='cpu', action='store_true')
    parser.add_argument('--dump', help='dump model and exit', dest='dump', action='store_true')
    
    opts, args = parser.parse_known_args()

    from modelmanager import ModelConfig

    if not os.path.exists(opts.config):
        ModelConfig(os.path.basename(opts.config).rsplit('.')[0]).save_config()
        exit()

    config = ModelConfig()
    config.load(opts.config)
    if opts.cpu:
        with tf.device('/cpu:0'):
            main(config, opts)
    else:
        main(config, opts)
