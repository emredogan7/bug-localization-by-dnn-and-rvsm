from util import csv2dict, tsv2dict
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold

samples_ = csv2dict("../data/features.csv")
samples = []

# oversampling
for i, sample in enumerate(samples_):
    samples.append(sample)
    if i % 51 in [0]:
        for _ in range(9):
            samples.append(sample)

np.random.shuffle(samples)

x_ = np.zeros((len(samples), 5))
y_ = np.zeros((len(samples), 1))

for i, sample in enumerate(samples):
    x_[i][0] = float(sample["rVSM_similarity"])
    x_[i][1] = float(sample["collab_filter"])
    x_[i][2] = float(sample["classname_similarity"])
    x_[i][3] = float(sample["bug_recency"])
    x_[i][4] = float(sample["bug_frequency"])
    y_[i] = float(sample["match"])


data_train, data_test, labels_train, labels_test = train_test_split(
    x_, y_, test_size=0.20, random_state=42
)

X = data_train
y = labels_train

# K-fold Cross Validation
kf = KFold(n_splits=10)
experiment_counter = 0
for train_index, test_index in kf.split(X):
    experiment_counter += 1
    print("Experiment Counter", experiment_counter)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )
    clf.fit(X_train, y_train.ravel())

    predicted = clf.predict(X_test)

    sample_dict = {}

    for sample in samples_:
        sample_dict[sample["report_id"]] = []

    for sample in samples_:
        temp_dict = {}
        temp_dict[sample["file"]] = [
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

    for i, counter in enumerate(topk_counters):
        acc = counter / (len(bug_reports) - negative_total)
        print("Accuracy of top", i + 1, ":", acc)


# Testing
for hidden_node_count in range(100, 1001, 100):
    print("Hidden node count", hidden_node_count)
    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(hidden_node_count,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )
    clf.fit(X, y.ravel())

    # predicted = clf.predict(data_test)

    sample_dict = {}

    for sample in samples_:
        sample_dict[sample["report_id"]] = []

    for sample in samples_:
        temp_dict = {}
        temp_dict[sample["file"]] = [
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

    for i, counter in enumerate(topk_counters):
        acc = counter / (len(bug_reports) - negative_total)
        print("Accuracy of top", i + 1, ":", acc)
