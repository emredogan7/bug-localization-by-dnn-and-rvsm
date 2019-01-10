from util import *
import os


bug_reports = tsv2dict()[1199:]
samples = csv2dict()

our_features_path = os.path.normpath('./data/our_features.csv')
with open(our_features_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["report_id", "file", "rVSM_similarity", "collab_filter", "classname_similarity", "bug_recency", "bug_frequency", "match"])

for bug_report in bug_reports:

    id = bug_report["id"]
    bug_id = bug_report["bug_id"]
    summary = bug_report["summary"]
    description = bug_report["description"]
    report_time = bug_report["report_time"]
    report_timestamp = bug_report["report_timestamp"]
    status = bug_report["status"]
    commit = bug_report["commit"]
    commit_timestamp = bug_report["commit_timestamp"]
    files = bug_report["files"]

    rawCorpus = bug_report["rawCorpus"]

    git_checkout(commit)
    javaFiles = getAllCorpus()

    date = convertToDateTime(report_time)

    collaborativeFilterScore, bugFixingRecency_, bugFixingFrequency_ = None, None, None
    for buggy_src_file in files:
        buggy_src_file = os.path.normpath(buggy_src_file)
        try:
            src = javaFiles[buggy_src_file]
        except:
            continue

        # rVSM Text Similarity
        rVSMTextSimilarity = cosine_sim(bug_report["rawCorpus"], src)

        # Collaborative Filter Score
        prevReports = getPreviousReportByFilename(buggy_src_file, date, bug_reports)
        relatedCorpus = []
        for report in prevReports:
            relatedCorpus.append(report["rawCorpus"])
        relatedString = ' '.join(relatedCorpus)
        collaborativeFilterScore = cosine_sim(bug_report["rawCorpus"], relatedString)

        # Class Name Similarity
        rawClassNames = src.split(" class ")[1:]

        classNames = []
        for block in rawClassNames:
            classNames.append(block.split(' ')[0])
        classCorpus = ' '.join(classNames)
        classNameSimilarity = cosine_sim(bug_report["rawCorpus"], classCorpus)

        # Bug Fixing Recency
        mrReport = getMostRecentReport(buggy_src_file, convertToDateTime(bug_report["report_time"]), bug_reports)
        bugFixingRecency_ = bugFixingRecency(bug_report, mrReport)

        # Bug Fixing Frequency
        bugFixingFrequency_ = bugFixingFrequency(buggy_src_file, date, bug_reports)

        our_features_path = os.path.normpath('./data/our_features.csv')
        with open(our_features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([id, buggy_src_file, rVSMTextSimilarity, collaborativeFilterScore, classNameSimilarity, bugFixingRecency_, bugFixingFrequency_, 1])

    for src_file in get_top_k_wrong_files(files, rawCorpus, javaFiles):
        with open(our_features_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([id, src_file[0], src_file[1], collaborativeFilterScore, src_file[2], bugFixingRecency_, bugFixingFrequency_, 0])
print()

# for bug_report_id, commit_id, description_terms, buggy_src_files:
#     pass

# commit_id = -1
# description_terms = []
# buggy_src_files = []
# for sample in samples:
#     report_id = sample["report_id"]
#     src_filename = sample["buggy_src_file"]
#     commit_id, description_terms, buggy_src_files = bug_report_dict[report_id]
#     git_checkout(commit_id)

#     src_filename = os.path.normpath('./data/eclipse.platform.ui/' + src_filename)
#     with open(src_filename) as f:
#         src_text = f.read()
#     print()
