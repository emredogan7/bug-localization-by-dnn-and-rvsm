from util import csv2dict, tsv2dict
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from dnn_model import helper_collections, topk_accuarcy


def rsvm_model():
    samples = csv2dict("../data/features.csv")
    rvsm_list = [float(sample["rVSM_similarity"]) for sample in samples]

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, bug_reports_files_dict = helper_collections(samples, True)

    acc_dict = topk_accuarcy(bug_reports, sample_dict, bug_reports_files_dict)

    return acc_dict


print(rsvm_model())
