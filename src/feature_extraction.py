""" 
A script for feature extraction.

Used many modified and intact code blocks from 
'https://github.com/jeffrey-palmerino/BugLocalizationDNN'
"""

from util import *
import csv
import os


def extract_features():
    # Clone git repo to a local folder
    git_clone(repo_url="https://github.com/eclipse/eclipse.platform.ui.git",
              clone_folder="../data/")

    # Read bug reports from tab separated file.
    bug_reports = tsv2dict('../data/Eclipse_Platform_UI.txt')

    features_path = os.path.normpath('../data/features.csv')
    with open(features_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["report_id", "file", "rVSM_similarity", "collab_filter",
                         "classname_similarity", "bug_recency", "bug_frequency", "match"])

    skipped_files_count = 0
    len_bug_reports = len(bug_reports)

    for i, br in enumerate(bug_reports):

        print("Bug report : {} / {}".format(i+1, len_bug_reports), end="\r")

        br_id = br["id"]
        br_date = br["report_time"]
        br_files = br["files"]
        br_raw_text = br["raw_text"]

        java_src_dict = get_all_source_code(
            "../data/eclipse.platform.ui/bundles/")

        cfs = None
        bfr = None
        bff = None

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
                prev_reports = previous_reports(
                    java_file, br_date, bug_reports)

                # Collaborative Filter Score
                cfs = collaborative_filtering_score(br_raw_text, prev_reports)

                # Bug Fixing Recency
                bfr = bug_fixing_recency(br, prev_reports)

                # Bug Fixing Frequency
                bff = len(prev_reports)

                features_path = os.path.normpath('../data/features.csv')
                with open(features_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [br_id, java_file, rvsm, cfs, cns, bfr, bff, 1])
            except:
                skipped_files_count += 1

        for java_file, rvsm, cns in top_k_wrong_files(br_files, br_raw_text, java_src_dict):
            with open(features_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [br_id, java_file, rvsm, cfs, cns, bfr, bff, 0])

    print("\n{} files are skipped.".format(skipped_files_count))


with CodeTimer("Feature extraction"):
    extract_features()
