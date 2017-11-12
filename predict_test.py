#!/bin/python3

import yaml
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import load_model

from train import conll_eval_counts, extract_features, extract_labels, print_cm, ordered_label_keys
import conlleval as ceval


def predict_test_file(fname, input_dim, timesteps, nlabels, labels):
    print('loading data from file ', fname)
    df = pd.read_csv(fname, sep=' ', header=0)
    X = extract_features(df, timesteps, input_dim)
    y = extract_labels(df, timesteps, nlabels)

    print('X temporal reshape: ', X.shape)
    print('y temporal reshape: ', y.shape)
    print('#samples: ', len(X))
    print('#labels: ', len(y))

    # we are averaging over all models output probabilities and then just taking the max
    m_preds = np.zeros((X.shape[0], timesteps, nlabels))
    for model in models:
        m_preds = m_preds + model.predict(X)
        break

    m_preds = m_preds / len(models)

    # just count and report and we are done
    counts, conf_matrix = conll_eval_counts(m_preds, y, labels)
    print('file: ', fname)
    ceval.report(counts)
    print_cm(conf_matrix, ordered_label_keys(labels))


# read all the keras models from the CV as an ensemble
models = []
for path in glob.glob('models/model*.hdf5'):
    print('loading ', path)
    models.append(load_model(path))

# TODO unify the duped code with train.py
input_dim = 50
timesteps = 20
nlabels = 8

cfg = yaml.load(open("data/meta.yaml", "r"))
if cfg['feature_dim']:
    input_dim = cfg['feature_dim']
if cfg['nlabels']:
    nlabels = cfg['nlabels']
if cfg['seq_len']:
    timesteps = cfg['seq_len']

labels = cfg['labels']

predict_test_file('data_test_a/vectorized.txt', input_dim, timesteps, nlabels, labels)
predict_test_file('data_test_b/vectorized.txt', input_dim, timesteps, nlabels, labels)
