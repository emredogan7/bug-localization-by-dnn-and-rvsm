from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def rsvm_model():
    samples = csv2dict("../data/features.csv")
    rvsm_list = [float(sample["rVSM_similarity"]) for sample in samples]

    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples, True)

    acc_dict = topk_accuarcy(bug_reports, sample_dict, br2files_dict)

    return acc_dict
