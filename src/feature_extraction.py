from util import *
import csv
import os

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


for i, br in enumerate(bug_reports):

    print("Bug repport : {} / {}".format(i+1, len(bug_reports)), end="\r")

    br_id = br["id"]
    br_date = br["report_time"]
    br_files = br["files"]
    br_raw_text = br["raw_text"]

    java_src_dict = get_all_source_code()

    collaborative_filter_score = None
    bfr = None
    bff = None

    for java_file in br_files:
        java_file = os.path.normpath(java_file)

        if java_file not in java_src_dict.keys():
            continue

        src = java_src_dict[java_file]

        # rVSM Text Similarity
        rvsm_sim = cosine_sim(br_raw_text, src)

        # Class Name Similarity
        classes = src.split(" class ")[1:]
        class_names = [c[:c.find(" ")] for c in classes]
        class_names_text = ' '.join(class_names)

        class_name_sim = cosine_sim(br_raw_text, class_names_text)

        # Previous reports
        prev_reports = previous_reports(java_file, br_date, bug_reports)

        # Collaborative Filter Score
        cfs = collaborative_filtering_score(br_raw_text, prev_reports)

        # Bug Fixing Recency
        bfr = bug_fixing_recency(br, prev_reports)

        # Bug Fixing Frequency
        bff = len(prev_reports)

        features_path = os.path.normpath('../data/features.csv')
        with open(features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([br_id, java_file, rvsm_sim, collaborative_filter_score,
                             class_name_sim, bfr, bff, 1])

    for src_file in get_top_k_wrong_files(br_files, br_raw_text, java_src_dict):
        with open(features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([br_id, src_file[0], src_file[1], collaborative_filter_score,
                             src_file[2], bfr, bff, 0])
