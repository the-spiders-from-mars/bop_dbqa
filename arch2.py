# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Separate hidden layers and convolution layers of questions and answers
"""
import logging
import pathlib

import jieba
import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, Dense, Activation, Conv1D, MaxPooling1D, concatenate, Flatten, merge, dot
from keras.preprocessing import sequence
from keras import backend as K

import qa_data
import w2v
import util.ProgressBar

__author__ = "freemso"

VERSION = "2.12"

TRAIN_DATA_FILE = "data/BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt"
DEV_DATA_FILE = "data/BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.dev.txt"
TEST_DATA_FILE = "data/BoP2017-DBQA.test.txt"

MODEL_WEIGHT_FILE = "model/arch_"+VERSION+".model"
DEV_OUTPUT_FILE = "out/dev_"+VERSION+".txt"
TEST_OUTPUT_FILE = "out/test_"+VERSION+".txt"

SENTENCE_LEN = 30
HIDDEN_LAYER_DIM = 200
NUM_FILTERS = 1000  # number of filters for each filter size
FILTER_SIZES = [2, 3, 4]  # kinds of filters
MARGIN = 0.009
LEARNING_RATE = 0.01

NUM_EPOCH = 10
BATCH_SIZE = 32

GAMMA = 1.0
C = 1


class Arch2(object):
    def __init__(self):
        self.w2v = w2v.WordVector()

        self.model = self._build()

        model_weight_file = pathlib.Path(MODEL_WEIGHT_FILE)
        if model_weight_file.is_file():
            logging.info("Loading model weight...")
            self.model.load_weights(MODEL_WEIGHT_FILE)
        else:
            logging.info("Training model...")
            # Train the model
            self._train(TRAIN_DATA_FILE)

    def test(self, file_path, out_file_path):
        import pickle
        # Preprocess the training data
        tuple_file = "temp/tuple_test"
        qpn_tuple_path = pathlib.Path(tuple_file)
        if qpn_tuple_path.is_file():
            logging.info("Loading preprocessed test data from file")
            with open(tuple_file, "rb") as in_file:
                questions, answers = pickle.load(in_file)
        else:
            logging.info("Test...")
            questions = []
            answers = []
            line_count = 0
            with open(file_path, encoding="utf-8-sig") as in_file:
                for _ in in_file:
                    line_count += 1
            bar = util.ProgressBar.ProgressBar(total=line_count)
            with open(file_path, encoding="utf-8-sig") as in_file:
                for line in in_file:
                    bar.log()
                    question, answer = line.strip().split("\t")
                    question_seg = jieba.cut(question)
                    question_word_idx = []
                    for word in question_seg:
                        if word in self.w2v.wv:
                            question_word_idx.append(self.w2v.word2idx[word] + 1)
                        else:
                            question_word_idx.append(0)
                    questions.append(question_word_idx)

                    answer_seg = jieba.cut(answer)
                    answer_word_idx = []
                    for word in answer_seg:
                        if word in self.w2v.wv:
                            answer_word_idx.append(self.w2v.word2idx[word] + 1)
                        else:
                            answer_word_idx.append(0)
                    answers.append(answer_word_idx)

            questions = sequence.pad_sequences(questions, maxlen=SENTENCE_LEN, padding="post", truncating="post",
                                               value=0)
            answers = sequence.pad_sequences(answers, maxlen=SENTENCE_LEN, padding="post", truncating="post", value=0)
            with open(tuple_file, "wb") as out_file:
                pickle.dump((questions, answers), out_file)

        # Predict
        sims, _ = self.model.predict({"question_input": questions,
                                      "answer_pos_input": answers,
                                      "answer_neg_input": np.zeros(answers.shape)})
        with open(out_file_path, "w") as out_file:
            for sim in sims:
                out_file.write(str(sim) + "\n")

    def predict(self, file_path, out_file_path):
        import pickle
        # Preprocess the training data
        tuple_file = "temp/tuple_predict"
        qpn_tuple_path = pathlib.Path(tuple_file)
        if qpn_tuple_path.is_file():
            logging.info("Loading preprocessed development data from file")
            with open(tuple_file, "rb") as in_file:
                questions, answers = pickle.load(in_file)
        else:
            logging.info("Predicting...")
            questions = []
            answers = []
            line_count = 0
            with open(file_path, encoding="utf-8-sig") as in_file:
                for _ in in_file:
                    line_count += 1
            bar = util.ProgressBar.ProgressBar(total=line_count)
            with open(file_path, encoding="utf-8-sig") as in_file:
                for line in in_file:
                    bar.log()
                    _, question, answer = line.strip().split("\t")
                    question_seg = jieba.cut(question)
                    question_word_idx = []
                    for word in question_seg:
                        if word in self.w2v.wv:
                            question_word_idx.append(self.w2v.word2idx[word] + 1)
                        else:
                            question_word_idx.append(0)
                    questions.append(question_word_idx)

                    answer_seg = jieba.cut(answer)
                    answer_word_idx = []
                    for word in answer_seg:
                        if word in self.w2v.wv:
                            answer_word_idx.append(self.w2v.word2idx[word] + 1)
                        else:
                            answer_word_idx.append(0)
                    answers.append(answer_word_idx)

            questions = sequence.pad_sequences(questions, maxlen=SENTENCE_LEN, padding="post", truncating="post", value=0)
            answers = sequence.pad_sequences(answers, maxlen=SENTENCE_LEN, padding="post", truncating="post", value=0)
            with open(tuple_file, "wb") as out_file:
                pickle.dump((questions, answers), out_file)

        # Predict
        sims, _ = self.model.predict({"question_input": questions,
                                      "answer_pos_input": answers,
                                      "answer_neg_input": np.zeros(answers.shape)})
        with open(out_file_path, "w") as out_file:
            for sim in sims:
                out_file.write(str(sim[0]) + "\n")

    def _train(self, file_path):
        import pickle
        # Preprocess the training data
        tuple_file = "temp/tuple"
        qpn_tuple_path = pathlib.Path(tuple_file)
        if qpn_tuple_path.is_file():
            logging.info("Loading preprocessed training data from file")
            with open(tuple_file, "rb") as in_file:
                questions, pos_answers, neg_answers = pickle.load(in_file)
        else:
            questions, pos_answers, neg_answers = self._preprocess(file_path)
            with open(tuple_file, "wb") as out_file:
                pickle.dump((questions, pos_answers, neg_answers), out_file)

        # Train model
        self.model.fit({"question_input": questions,
                        "answer_pos_input": pos_answers,
                        "answer_neg_input": neg_answers},
                       {"pos_sim_output": np.ones(questions.shape[0], dtype="float32"),
                        "neg_sim_output": -np.ones(questions.shape[0], dtype="float32")},
                       epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

        # save model weights
        self.model.save_weights(MODEL_WEIGHT_FILE, overwrite=True)

    def _build(self):
        """ To setup the structure of the network """
        # embedding layer weight setting
        import keras.backend.tensorflow_backend as K
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=config))
        embedding_weights = np.zeros((self.w2v.vocab_size + 1, self.w2v.vector_size))
        embedding_weights[0, :] = np.zeros((self.w2v.vector_size,))
        for index, word in self.w2v.idx2word.items():
            embedding_weights[index + 1, :] = self.w2v.wv[word]

        # Network structure
        logging.info("Building model...")
        embedding_layer = Embedding(self.w2v.vocab_size + 1, self.w2v.vector_size,
                                    weights=[embedding_weights], input_length=SENTENCE_LEN,
                                    trainable=True, name="emb")
        from keras import regularizers
        hidden_layer = Dense(HIDDEN_LAYER_DIM, use_bias=True, name="hidden", kernel_regularizer=regularizers.l2(0.0001))

        q_in = Input(shape=(SENTENCE_LEN,), name="question_input")
        a_pos_in = Input(shape=(SENTENCE_LEN,), name="answer_pos_input")
        a_neg_in = Input(shape=(SENTENCE_LEN,), name="answer_neg_input")

        q = embedding_layer(q_in)
        a_pos = embedding_layer(a_pos_in)
        a_neg = embedding_layer(a_neg_in)

        q = hidden_layer(q)
        a_pos = hidden_layer(a_pos)
        a_neg = hidden_layer(a_neg)

        concat_q = []
        concat_a_pos = []
        concat_a_neg = []
        for sz in FILTER_SIZES:
            cnn = Conv1D(NUM_FILTERS, sz, strides=1, name="conv_" + str(sz))

            pooling = MaxPooling1D(pool_size=SENTENCE_LEN - sz + 1, name="question_pool_" + str(sz))

            q_cnn = cnn(q)
            a_pos_cnn = cnn(a_pos)
            a_neg_cnn = cnn(a_neg)

            q_pool = pooling(q_cnn)
            a_pos_pool = pooling(a_pos_cnn)
            a_neg_pool = pooling(a_neg_cnn)

            concat_q.append(q_pool)
            concat_a_pos.append(a_pos_pool)
            concat_a_neg.append(a_neg_pool)
        q = Activation(activation="tanh", name="question_tanh")(Flatten()(concatenate(concat_q, axis=-1)))
        a_pos = Activation(activation="tanh", name="answer_pos_tanh")(Flatten()(concatenate(concat_a_pos, axis=-1)))
        a_neg = Activation(activation="tanh", name="answer_neg_tanh")(Flatten()(concatenate(concat_a_neg, axis=-1)))

        pos_sim = merge([q, a_pos], mode=dot_sim, output_shape=lambda x: x[:-1], name="pos_sim_output")
        neg_sim = merge([q, a_neg], mode=dot_sim, output_shape=lambda x: x[:-1], name="neg_sim_output")

        model = Model(inputs=[q_in, a_pos_in, a_neg_in], outputs=[pos_sim, neg_sim])

        # compile model
        model.compile(loss=max_margin_loss,
                      optimizer="sgd")

        return model

    def _preprocess(self, file_path):
        q2doc = {}
        with open(file_path, encoding="utf-8-sig") as in_file:
            for line in in_file:
                label, question, answer = line.strip().split("\t")
                if question not in q2doc.keys():
                    q2doc[question] = qa_data.Doc(question)
                q2doc[question].add_ans(answer, bool(int(label)))

        questions = []
        pos_answers = []
        neg_answers = []
        bar = util.ProgressBar.ProgressBar(total=len(q2doc.values()))
        for doc in q2doc.values():
            bar.log()
            question = doc.question

            question_seg = jieba.cut(question)
            question_word_idx = []
            for word in question_seg:
                if word in self.w2v.wv:
                    question_word_idx.append(self.w2v.word2idx[word] + 1)
                else:
                    question_word_idx.append(0)

            for pos_answer in doc.pos_answers:
                pos_answer_seg = jieba.cut(pos_answer)
                pos_answer_word_idx = []
                for word in pos_answer_seg:
                    if word in self.w2v.wv:
                        pos_answer_word_idx.append(self.w2v.word2idx[word] + 1)
                    else:
                        pos_answer_word_idx.append(0)

                for neg_answer in doc.neg_answers:
                    neg_answer_seg = jieba.cut(neg_answer)
                    neg_answer_word_idx = []
                    for word in neg_answer_seg:
                        if word in self.w2v.wv:
                            neg_answer_word_idx.append(self.w2v.word2idx[word] + 1)
                        else:
                            neg_answer_word_idx.append(0)
                    questions.append(question_word_idx)
                    pos_answers.append(pos_answer_word_idx)
                    neg_answers.append(neg_answer_word_idx)

        questions = sequence.pad_sequences(questions, maxlen=SENTENCE_LEN, padding="post", truncating="post", value=0)
        pos_answers = sequence.pad_sequences(pos_answers, maxlen=SENTENCE_LEN, padding="post", truncating="post",
                                             value=0)
        neg_answers = sequence.pad_sequences(neg_answers, maxlen=SENTENCE_LEN, padding="post", truncating="post",
                                             value=0)
        return questions, pos_answers, neg_answers


def max_margin_loss(y_true, y_pred):
    signed = y_pred * y_true  # we do this, just so that y_true is part of the computational graph
    pos = signed[0::2]
    neg = signed[1::2]
    # negative samples are multiplied by -1, so that the sign in the rankSVM objective is flipped below
    return K.max(K.maximum(0., MARGIN - pos - neg))


def dot_sim(x):
    return dot(x, axes=-1, normalize=True)


# def polynomial(x):
#     return (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
#
#
# def sigmoid(x):
#     return K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
#
#
# def rbf(x):vc
#     return K.exp(-1 * params['gamma'] * l2_norm(x[0] - x[1]) ** 2)
#
#
# def euclidean(x):
#     return 1 / (1 + l2_norm(x[0] - x[1]))
#
#
# def exponential(x):
#     return K.exp(-1 * params['gamma'] * l2_norm(x[0] - x[1]))
#
#
def gesd(x):
    x0 = K.l2_normalize(x[0], axis=-1)
    x1 = K.l2_normalize(x[1], axis=-1)
    euclidean = 1 / (1 + K.sqrt(K.batch_dot(x0 - x1, x0 - x1, axes=-1)))
    sigmoid = 1 / (1 + K.exp(-1 * GAMMA * (K.batch_dot(x0, x1, axes=-1) + C)))
    return euclidean * sigmoid
#
#
# def aesd(x):
#     euclidean = 0.5 / (1 + l2_norm(x[0] - x[1]))
#     sigmoid = 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
#     return euclidean + sigmoid


def main():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    arch2 = Arch2()
    arch2.predict(DEV_DATA_FILE, DEV_OUTPUT_FILE)
    arch2.test(TEST_DATA_FILE, TEST_OUTPUT_FILE)


if __name__ == '__main__':
    main()
