from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed, cpu_count
from math import ceil
import numpy as np
import os


def oversample(samples_):
    samples = []

    # oversample features of buggy files
    for i, sample in enumerate(samples_):
        samples.append(sample)
        if i % 51 == 0:
            for _ in range(9):
                samples.append(sample)

    return samples


def features_and_labels(samples):
    features = np.zeros((len(samples), 5))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        labels[i] = float(sample["match"])

    return features, labels


def kfold_split_indexes(k, len_samples):
    step = ceil(len_samples / k)
    ret_list = [(start, step) for start in range(0, len_samples, step)]

    return ret_list


def kfold_split(bug_reports, samples, start, step):
    train_samples = samples[:start] + samples[start + step :]
    test_samples = samples[start : start + step]

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports if br["id"] in test_br_ids]

    return train_samples, test_bug_reports


def train_dnn(
    i, num_folds, samples, start, step, sample_dict, bug_reports, br2files_dict
):
    print("Fold: {} / {}".format(i + 1, num_folds), end="\r")

    train_samples, test_bug_reports = kfold_split(bug_reports, samples, start, step)
    train_samples = oversample(train_samples)
    np.random.shuffle(train_samples)
    X_train, y_train = features_and_labels(train_samples)

    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )
    clf.fit(X_train, y_train.ravel())

    acc_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=clf)

    return acc_dict


def dnn_model_kfold(k=10):
    samples = csv2dict("../data/features.csv")

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples)

    np.random.shuffle(samples)

    # K-fold Cross Validation in parallel 
    acc_dicts = Parallel(n_jobs=-2)( # Uses all cores but one
        delayed(train_dnn)(
            i, k, samples, start, step, sample_dict, bug_reports, br2files_dict
        )
        for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    )

    # Calculating the average accuracy from all folds
    avg_acc_dict = {}
    for key in acc_dicts[0].keys():
        avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)

    return avg_acc_dict
