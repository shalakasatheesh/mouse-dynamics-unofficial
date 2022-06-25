import argparse
from move_dataset import MoveDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from keras_utils import get_model_output_features, truncate_model
from sklearn.svm import OneClassSVM
from sklearn import metrics
import numpy as np
from random import uniform
import pandas as pd

dataset = MoveDataset(pickle_file='move_data.pkl')

X, y_onehot = dataset.train_data()
X = preprocessing.StandardScaler().fit_transform(X.reshape(-1, 256)).reshape(-1, 128, 2)

# One hot vector to single value vector
y = np.argmax(y_onehot, axis=1)

# Source: https://github.com/margitantal68/sapimouse/blob/40b5ea6cf10c6f1d64b9dd0427d21138cc4f75e2/util/oneclass.py#L40
def compute_AUC_EER(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)
    return roc_auc, EER

# Source: https://github.com/margitantal68/sapimouse/blob/40b5ea6cf10c6f1d64b9dd0427d21138cc4f75e2/util/oneclass.py#L138
def score_normalization(positive_scores, negative_scores):
    scores = [positive_scores, negative_scores]
    scores_df = pd.DataFrame(scores)

    mean = scores_df.mean()
    std = scores_df.std()
    min_score = mean - 2 * std
    max_score = mean + 2 * std

    min_score = min_score[0]
    max_score = max_score[0]

    positive_scores = [(x - min_score) / (max_score - min_score) for x in positive_scores]
    positive_scores = [(uniform(0.0, 0.05) if x < 0 else x) for x in positive_scores]
    positive_scores = [(uniform(0.95, 1.0) if x > 1 else x) for x in positive_scores]

    negative_scores = [(x - min_score) / (max_score - min_score) for x in negative_scores]
    negative_scores = [uniform(0.0, 0.05) if x < 0 else x for x in negative_scores]
    negative_scores = [uniform(0.95, 1.0) if x > 1 else x for x in negative_scores]
    return positive_scores, negative_scores


if __name__ == "__main__":
    # Extract parameters from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_trained.h5")
    args = parser.parse_args()

    model = truncate_model(tf.keras.models.load_model(args.model))

    for userid in range(dataset.unique_user_count()):
        if np.count_nonzero(y == userid) == 0:
            continue
        X_positive = X[y == userid]
        X_negative = X[y != userid]

        # Extract new features from dx, dy vectors
        X_positive_features = get_model_output_features(model, X_positive)
        X_negative_features = get_model_output_features(model, X_negative)

        clf = OneClassSVM(gamma='scale')
        clf.fit(X_positive_features)

        positive_scores = clf.score_samples(X_positive_features)
        negative_scores = clf.score_samples(X_negative_features)

        auc, eer = compute_AUC_EER(positive_scores, negative_scores)

        positive_scores, negative_scores = score_normalization(positive_scores, negative_scores)
        print('auc:', auc, 'eer:', eer)
