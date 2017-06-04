# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Chinese word embedding model
"""

import logging
import multiprocessing
import pathlib

import gensim.models
import jieba
#import opencc
import gensim.models.word2vec

from util import preprocessing

__author__ = "freemso"

MODEL_PATH = "model/word2vec.model"
DATA_PATH = "data/sougoCA/sougoCA_txt.data"
RAW_DATA_PATH = "data/sougoCA/sougoCA_xml.data"
CHAR_VEC_DIM = 200


class WordVector:
    def __init__(self):
        model_file_path = pathlib.Path(MODEL_PATH)
        if model_file_path.is_file():
            # load model from disk file
            logging.info("Loading Char2Vec model from disk...")
            self._model = gensim.models.Word2Vec.load(MODEL_PATH)
        else:
            logging.info("Training Char2Vec model...")
            sougo_file = pathlib.Path(DATA_PATH)
            if not sougo_file.is_file():
                load_sougo_data()
            self._model = gensim.models.Word2Vec(gensim.models.word2vec.LineSentence(DATA_PATH),
                                                 sg=1, size=CHAR_VEC_DIM, min_count=5,
                                                 workers=multiprocessing.cpu_count())
            # save model to disk
            self._model.save(MODEL_PATH)

        self.wv = self._model.wv
        self.word2idx = dict((w, i) for i, w in enumerate(self._model.wv.index2word))
        self.idx2word = dict((i, w) for i, w in enumerate(self._model.wv.index2word))
        self.vocab_size = len(self._model.wv.index2word)
        self.vector_size = self._model.vector_size


def load_sougo_data():
    pass
# def load_sougo_data():
#     logging.info("Loading sougo data...")
#     # count = 0
#     with open(RAW_DATA_PATH) as in_file, open(DATA_PATH, "w") as out_file:
#         cc = opencc.OpenCC(config="t2s.json")
#         for line in in_file:
#             # count += 1
#             # if count > 1000:
#             #     break
#             if line.strip().startswith("<content>"):
#                 txt = line.strip("</content>")
#                 if txt is not None:
#                     # convert to simplified chinese
#                     sentence = cc.convert(txt)
#                     sent_seg = jieba.cut(sentence)
#                     line = " ".join([word for word in sent_seg])
#                     out_file.write(line + "\n")


def main():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    WordVector()


if __name__ == '__main__':
    main()
