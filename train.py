#!/bin/python3

import numpy as np
import pandas as pd
import yaml
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, Convolution1D, LSTM
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dropout, Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Sequential, load_model
from keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import conlleval as ceval


def extract_features(df, timesteps, input_dim):
    X = df.iloc[:, timesteps:].values
    return X.reshape(X.shape[0], -1, input_dim).astype('float32')


def extract_labels(df, timesteps, nlabels):
    y = df.iloc[:, 0:timesteps]
    yx = []
    for index, row in y.iterrows():
        yx.append(to_categorical(row, nlabels))

    return np.array(yx)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes, from https://gist.github.com/zachguo/10296432"""
    print()
    columnwidth = max([len(x) for x in labels] + [7])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
    print()


def ordered_label_keys(labels):
    return list(map(lambda x: x[1], sorted(labels.items())))


def conll_eval_counts(ypred, ytruth, labels):
    ytruth_max = ytruth.argmax(axis=2)
    ypred_max = ypred.argmax(axis=2)
    conf_matrix = None
    eval_counts = ceval.EvalCounts()
    label_keys = ordered_label_keys(labels)
    for i in range(len(ypred_max)):
        true_seq = [labels[x] for x in ytruth_max[i].tolist()]
        pred_seq = [labels[x] for x in ypred_max[i].tolist()]
        c = ceval.evaluate(['%s %s' % x for x in zip(true_seq, pred_seq)])
        eval_counts.add(c)

        cm = confusion_matrix(true_seq, pred_seq, label_keys)
        conf_matrix = cm if conf_matrix is None else conf_matrix + cm

    return eval_counts, conf_matrix


class ConllEvaluation(Callback):
    def __init__(self, prefix, model, test_features, test_ground_truth, labels):
        super().__init__()
        self.prefix = prefix
        self.model = model
        self.test_features = test_features
        self.test_ground_truth = test_ground_truth
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        ypred = self.model.predict(self.test_features)
        c, cmat = conll_eval_counts(ypred, self.test_ground_truth, self.labels)
        ceval.report(c, prefix=self.prefix)
        print_cm(cmat, ordered_label_keys(self.labels))
        o, b = ceval.metrics(c)

        # tensorboard requires those logs to be float64 with attribute item(), thus we create them with numpy
        logs[self.prefix + "_conll_f1"] = np.float64(o.fscore)
        logs[self.prefix + "_conll_prec"] = np.float64(o.prec)
        logs[self.prefix + "_conll_rec"] = np.float64(o.rec)


if __name__ == '__main__':
    # training
    nfolds = 5
    nb_epoch = 400
    batch_size = 128
    nlabels = 8

    # conv
    nb_filter = 512
    kernel_size = 1
    strides = 1
    # neither causal nor same seem to work very well here
    padding = 'valid'
    dilation = 1

    # Recurrent
    timesteps = 10
    input_dim = 50
    rec_dim = 150

    cfg = yaml.load(open("data/meta.yaml", "r"))
    if cfg['seq_len']:
        timesteps = cfg['seq_len']
    if cfg['feature_dim']:
        input_dim = cfg['feature_dim']
    if cfg['nlabels']:
        nlabels = cfg['nlabels']

    labels = cfg['labels']

    print('timesteps: {}, input dim: {}, num output labels: {}'.format(timesteps, input_dim, nlabels))


    def nn_model():
        m = Sequential()

        # after a bunch of iterations the weights went straight towards a normal distribution centered along zero
        # with a little stddev of around 1.5-0.2, so we initialize these directly from that knowledge
        init = RandomNormal(mean=0, stddev=0.2)

        m.add(Convolution1D(filters=nb_filter,
                            kernel_size=kernel_size,
                            kernel_initializer=init,
                            strides=strides,
                            padding=padding,
                            dilation_rate=dilation,
                            input_shape=(timesteps, input_dim)))
        m.add(BatchNormalization())
        m.add(PReLU())
        m.add(Dropout(0.5))

        m.add(Convolution1D(filters=nb_filter,
                            kernel_size=kernel_size,
                            kernel_initializer=init,
                            strides=strides,
                            dilation_rate=dilation,
                            padding=padding))
        m.add(BatchNormalization())
        m.add(PReLU())
        m.add(Dropout(0.5))

        m.add(Bidirectional(LSTM(rec_dim, kernel_initializer=init, return_sequences=True)))
        m.add(BatchNormalization())
        m.add(PReLU())
        m.add(Dropout(0.3))

        m.add(TimeDistributed(Dense(nlabels, activation='softmax', kernel_initializer='he_normal')))

        opt = Adadelta(clipvalue=1.0)
        m.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_accuracy'])
        return m


    # fail fast if a model does not compile
    nn_model().summary()

    df = pd.read_csv('data/vectorized.txt', sep=' ', header=0)
    X = extract_features(df, timesteps, input_dim)
    y = extract_labels(df, timesteps, nlabels)

    print('X temporal reshape: ', X.shape)
    print('y temporal reshape: ', y.shape)
    print('#example sequences: ', len(X))
    print('#label sequences: ', len(y))

    df_valid = pd.read_csv('data_test_a/vectorized.txt', sep=' ', header=0)
    X_valid = extract_features(df_valid, timesteps, input_dim)
    y_valid = extract_labels(df_valid, timesteps, nlabels)

    folds = KFold(n_splits=nfolds, shuffle=True)
    currentFold = 0
    foldScores = []
    for (inTrain, inTest) in folds.split(X):
        xtr = X[inTrain]
        ytr = y[inTrain]
        xte = X[inTest]
        yte = y[inTest]

        print('Fold ', currentFold, ' starting...')
        checkPointPath = 'models/model_fold_{}.hdf5'.format(currentFold)
        model = nn_model()
        callbacks = [
            ConllEvaluation(prefix='test', model=model, test_features=xte, test_ground_truth=yte, labels=labels),
            ConllEvaluation(prefix='valid', model=model, test_features=X_valid, test_ground_truth=y_valid,
                            labels=labels),
            EarlyStopping(monitor='valid_conll_f1', patience=20, verbose=0, mode='max'),
            ModelCheckpoint(monitor='valid_conll_f1', filepath=checkPointPath,
                            verbose=0, save_best_only=True, mode='max'),
            TensorBoard(log_dir='./logs/fold_{}/'.format(currentFold), histogram_freq=1)
        ]

        model.fit(xtr, ytr, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1, validation_data=(xte, yte),
                  callbacks=callbacks)

        print('loading the currently best model for final evaluation...')
        model = load_model(checkPointPath)

        print('--------------------------------------------------')
        print('Fold ', currentFold, ' performance')
        counts, cmat = conll_eval_counts(model.predict(xte), yte, labels)
        overall, byType = ceval.metrics(counts)
        ceval.report(counts)
        print_cm(cmat, ordered_label_keys(labels))
        foldScores.append(overall.fscore)
        print('\n')
        print('avg f1 fold scores so far: ', np.mean(foldScores))
        currentFold += 1

        # we clear the tensorflow session after each fold to not leak resources
        K.clear_session()

    print('f1 fold scores: ', foldScores)
    print('final avg f1 fold scores: ', np.mean(foldScores))
