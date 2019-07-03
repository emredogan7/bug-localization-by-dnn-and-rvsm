from util import *
import csv
import os


git_clone()

bug_reports = tsv2dict()
# samples = csv2dict()

features_path = os.path.normpath('../data/features.csv')
with open(features_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["report_id", "file", "rVSM_similarity", "collab_filter",
                     "classname_similarity", "bug_recency", "bug_frequency", "match"])

for bug_report in bug_reports:

    report_id = bug_report["id"]
    bug_id = bug_report["bug_id"]
    summary = bug_report["summary"]
    description = bug_report["description"]
    report_date = bug_report["report_time"]
    report_timestamp = bug_report["report_timestamp"]
    status = bug_report["status"]
    commit = bug_report["commit"]
    commit_timestamp = bug_report["commit_timestamp"]
    files = bug_report["files"]
    raw_corpus = bug_report["raw_text"]

    java_file_dict = get_all_source_code()

    collaborative_filter_score = None
    bug_fixing_recency_ = None
    bug_fixing_frequency_ = None

    for buggy_file in files:
        buggy_file = os.path.normpath(buggy_file)

        if buggy_file not in java_file_dict.keys():
            continue

        src = java_file_dict[buggy_file]

        # rVSM Text Similarity
        rvsm_text_sim = cosine_sim(raw_corpus, src)

        # Collaborative Filter Score
        prev_reports = previous_reports(
            buggy_file, report_date, bug_reports)
        prev_reports_combined_text = ""
        for report in prev_reports:
            prev_reports_combined_text += report["raw_text"]
        
        collaborative_filter_score = cosine_sim(
            raw_corpus, prev_reports_combined_text)

        # Class Name Similarity
        raw_class_names = src.split(" class ")[1:]

        class_names = []
        for block in raw_class_names:
            class_names.append(block.split(' ')[0])
        class_corpus = ' '.join(class_names)
        class_name_sim = cosine_sim(raw_corpus, class_corpus)

        # Bug Fixing Recency
        most_recent_report = get_most_recent_report(
            buggy_file, report_date, bug_reports)
        bug_fixing_recency_ = bug_fixing_recency(
            bug_report, most_recent_report)

        # Bug Fixing Frequency
        bug_fixing_frequency_ = bug_fixing_frequency(
            buggy_file, report_date, bug_reports)

        features_path = os.path.normpath('../data/features.csv')
        with open(features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([report_id, buggy_file, rvsm_text_sim, collaborative_filter_score,
                             class_name_sim, bug_fixing_recency_, bug_fixing_frequency_, 1])

    for src_file in get_top_k_wrong_files(files, raw_corpus, java_file_dict):
        with open(features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([report_id, src_file[0], src_file[1], collaborative_filter_score,
                             src_file[2], bug_fixing_recency_, bug_fixing_frequency_, 0])
