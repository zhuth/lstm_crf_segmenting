import re
import os
import math
import glob
import numpy as np
from itertools import chain
import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class DataSet:
    
    RANDOM_STATE = 40

    def __init__(self, filename='', data_size=1, data_x=None, data_y=None, word2id={}, id2word={}, tag2id={}, id2tag=[]):
        if filename:
            with open(filename, 'rb') as f:
                self.data_x = pickle.load(f)
                self.data_y = pickle.load(f)
                self.word2id = pickle.load(f)
                self.id2word = pickle.load(f)
                self.tag2id = pickle.load(f)
                self.id2tag = pickle.load(f)
            count = int(len(self.data_x) * data_size)
            self.data_x, self.data_y = self.data_x[:count], self.data_y[:count]
        else:
            self.data_x, self.data_y = data_x, data_y
            self.word2id = word2id
            self.id2word = id2word
            self.tag2id = tag2id
            self.id2tag = id2tag
        self.split_data()

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data_x, f)
            pickle.dump(self.data_y, f)
            pickle.dump(self.word2id, f)
            pickle.dump(self.id2word, f)
            pickle.dump(self.tag2id, f)
            pickle.dump(self.id2tag, f)

    def split_data(self, test_size=0.2):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=test_size, random_state=DataSet.RANDOM_STATE)

    @property
    def n_words(self):
        return len(self.id2word) + 1
    
    @property
    def n_tags(self):
        return len(self.id2tag)

    @property
    def n_docs(self):
        return len(self.data_x)

    @property
    def doc_length(self):
        return len(self.data_x[0])

    def visualize(self, x, y, delimiter='', join=''):
        return join.join([self.id2word[w]+delimiter+self.id2tag[t] for w, t in zip(x, y) if w])


class DataPreparer:

    def __init__(self, biaodian, max_length, filename):
        self.biaodian = [''] + biaodian
        self.filename = filename
        self.max_length = max_length

    def prepare(self, text_file='', limit=-1):

        def prepare_paragraphs(fin):
            paragraph = ''
            for l in fin:
                if l.startswith('   '):
                    yield paragraph
                    paragraph = ''
                paragraph += l.strip()
            yield paragraph

        def prepare_labels(paragraphs):
            import zhon.hanzi
            for p in paragraphs:
                if not p or len(p) < 10: continue
                p = re.sub(r'[（）—“”〈〉【】]|[^\u2e80-\u9fff\ufb00-\ufffd]', '', p)
                p = re.sub(r'[%s]' % ''.join([_ for _ in zhon.hanzi.punctuation if _ not in self.biaodian]), '。', p)
                sentences = re.split('(' + '|'.join(self.biaodian[1:]) + ')', p)
                label = []
                ptext = ''
                for s, bd in zip(sentences[::2], sentences[1::2]):
                    ptext += s
                    label += [0]*(len(s)-1) + [self.biaodian.index(bd)]
                if ptext and sum(label):
                    yield list(ptext), label

        words, labels = [], []
        print('Start creating words and labels...')
        
        paragraphs = prepare_paragraphs(open(text_file, glob.glob('*.txt')[0], errors='ignore', encoding='gbk'))
        for _i, (p, l) in enumerate(prepare_labels(paragraphs)):
            if _i == limit: break
            words.append(np.asarray(p))
            labels.append(np.asarray(l))

        print('Words Length', len(words), 'Labels Length', len(labels))
        print('Words Example', words[10])
        print('Labels Example', labels[10])
        id2word = list(set(chain(*words)))

        word2id = {w: i for i, w in enumerate(id2word)}
        tag2id = {t: i for i, t in enumerate(self.biaodian)}

        print('Starting transform...')
        
        data_x = pad_sequences(maxlen=self.max_length, sequences=list(map(lambda w: [word2id[_] for _ in w], words)), padding="post", value=0)
        data_y = pad_sequences(maxlen=self.max_length, sequences=labels, padding="post", value=0)

        print('Saving to', self.filename, '...')
        DataDump(data_x=data_x, data_y=data_y, word2id=word2id, id2word=id2word, tag2id=tag2id, id2tag=self.biaodian).dump(self.filename)
    