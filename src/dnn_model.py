from util import csv2dict, tsv2dict
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed, cpu_count
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


def some_collections(samples_):
    sample_dict = {}
    for sample in samples_:
        sample_dict[sample["report_id"]] = []

    for sample in samples_:
        temp_dict = {}
        temp_dict[os.path.normpath(sample["file"])] = [
            float(sample["rVSM_similarity"]),
            float(sample["collab_filter"]),
            float(sample["classname_similarity"]),
            float(sample["bug_recency"]),
            float(sample["bug_frequency"]),
        ]

        sample_dict[sample["report_id"]].append(temp_dict)

    bug_reports = tsv2dict("../data/Eclipse_Platform_UI.txt")
    bug_reports_files_dict = {}

    for bug_report in bug_reports:
        bug_id = bug_report["id"]
        bug_reports_files_dict[bug_id] = bug_report["files"]

    return sample_dict, bug_reports, bug_reports_files_dict


def train_dnn(
    i,
    features_train,
    labels_train,
    train_index,
    test_index,
    sample_dict,
    bug_reports,
    bug_reports_files_dict,
):
    print("Experiment: {} / {}".format(i + 1, num_folds), end="\r")
    X_train, X_test = features_train[train_index], features_train[test_index]
    y_train, y_test = labels_train[train_index], labels_train[test_index]

    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )
    clf.fit(X_train, y_train.ravel())

    # predicted = clf.predict(X_test)

    topk_counters = [0] * 20
    negative_total = 0
    for bug_report in bug_reports:
        dnn_input = []
        corresponding_files = []
        bug_id = bug_report["id"]

        try:
            for temp_dict in sample_dict[bug_id]:
                key = list(temp_dict.keys())[0]
                value = list(temp_dict.values())[0]
                dnn_input.append(value)
                corresponding_files.append(key)
        except:
            negative_total += 1
            continue

        relevancy_list = clf.predict(dnn_input)

        for i in range(1, 21):
            max_indices = np.argpartition(relevancy_list, -i)[-i:]
            for corresponding_file in np.array(corresponding_files)[max_indices]:
                if str(corresponding_file) in bug_reports_files_dict[bug_id]:
                    topk_counters[i - 1] += 1
                    break

    acc_dict = {}
    for i, counter in enumerate(topk_counters):
        acc = counter / (len(bug_reports) - negative_total)
        acc_dict[i + 1] = acc
        # print("Accuracy of top", i + 1, ":", acc)

    return acc_dict


samples_ = csv2dict("../data/features.csv")

sample_dict, bug_reports, bug_reports_files_dict = some_collections(samples_)

samples = oversample(samples_)
np.random.shuffle(samples)
features, labels = features_and_labels(samples)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.20, random_state=42
)

# K-fold Cross Validation
num_folds = 10
kf = KFold(n_splits=num_folds)

acc_dicts = Parallel(n_jobs=cpu_count() - 1)(
    delayed(train_dnn)(
        i,
        features_train,
        labels_train,
        train_index,
        test_index,
        sample_dict,
        bug_reports,
        bug_reports_files_dict,
    )
    for i, (train_index, test_index) in enumerate(kf.split(features_train))
)

print(acc_dicts)
