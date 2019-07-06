""" 
A script for feature extraction.

Used many modified and intact code blocks from 
'https://github.com/jeffrey-palmerino/BugLocalizationDNN'
"""

from util import *
from joblib import Parallel, delayed, cpu_count
import csv
import os


def extract(i, br, bug_reports):
    """Extracts 1 right and 50 wrong features for the given bug report
    
    Arguments:
        i {integer} -- Index for printing information
        br {dictionary} -- Given bug report 
        bug_reports {list of dictionaries} -- All bug reports
    """
    print("Bug report : {} / {}".format(i + 1, len(bug_reports)), end="\r")

    br_id = br["id"]
    br_date = br["report_time"]
    br_files = br["files"]
    br_raw_text = br["raw_text"]

    java_src_dict = get_all_source_code("../data/eclipse.platform.ui/bundles/")

    cfs = None
    bfr = None
    bff = None
    features = []

    for java_file in br_files:
        java_file = os.path.normpath(java_file)

        try:
            # Source code of the java file
            src = java_src_dict[java_file]

            # rVSM Text Similarity
            rvsm = cosine_sim(br_raw_text, src)

            # Class Name Similarity
            cns = class_name_similarity(br_raw_text, src)

            # Previous Reports
            prev_reports = previous_reports(java_file, br_date, bug_reports)

            # Collaborative Filter Score
            cfs = collaborative_filtering_score(br_raw_text, prev_reports)

            # Bug Fixing Recency
            bfr = bug_fixing_recency(br, prev_reports)

            # Bug Fixing Frequency
            bff = len(prev_reports)

            features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, 1])
        except:
            pass

    for java_file, rvsm, cns in top_k_wrong_files(br_files, br_raw_text, java_src_dict):
        features.append([br_id, java_file, rvsm, cfs, cns, bfr, bff, 0])

    return features


def extract_features():
    """Clones the git repository and parallelizes the feature extraction process
    """
    # Clone git repo to a local folder
    git_clone(
        repo_url="https://github.com/eclipse/eclipse.platform.ui.git",
        clone_folder="../data/",
    )

    # Read bug reports from tab separated file
    bug_reports = tsv2dict("../data/Eclipse_Platform_UI.txt")

    # Use all CPUs except one to speed up extraction and avoid computer lagging
    batches = Parallel(n_jobs=cpu_count() - 1)(
        delayed(extract)(i, br, bug_reports) for i, br in enumerate(bug_reports)
    )

    # Flatten features
    features = [row for batch in batches for row in batch]

    # Save features to a csv file
    features_path = os.path.normpath("../data/features.csv")
    with open(features_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "report_id",
                "file",
                "rVSM_similarity",
                "collab_filter",
                "classname_similarity",
                "bug_recency",
                "bug_frequency",
                "match",
            ]
        )
        for row in features:
            writer.writerow(row)


# Keep time while extracting features
with CodeTimer("Feature extraction"):
    extract_features()
