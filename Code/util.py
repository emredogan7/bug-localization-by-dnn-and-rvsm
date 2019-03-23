import csv
import re
import os
import random
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def tsv2dict():
    file_ = os.path.normpath('./data/Eclipse_Platform_UI.txt')
    reader = csv.DictReader(open(file_, 'r'), delimiter='\t')
    dict_list = []
    for line in reader:
        line["files"] = [
            s.strip() + ".java" for s in line["files"].split(".java") if s]
        line["rawCorpus"] = line["summary"][10:] + line["description"]
        line["summary"], line["description"], combined_corpus = get_combined_corpus(
            line)

        dict_list.append(line)
    return dict_list


def get_combined_corpus(report):
    report["summary"] = clean_and_split(report["summary"])[2:]
    report["description"] = clean_and_split(report["description"])
    combined_corpus = report["summary"] + report["description"]
    return report["summary"], report["description"], combined_corpus


def clean_and_split(text):

    return_text = re.sub(r'[^\w\s]', '', text)

    return_text = [s.strip() for s in return_text.split()]
    return return_text


def csv2dict():
    file_ = os.path.normpath('./data/features.csv')
    with open(file_, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict


def git_checkout(commmit_id):
    eclipse_ui_git_directory = os.path.normpath("./data/eclipse.platform.ui")
    os.chdir(eclipse_ui_git_directory)

    next_commit_id = commmit_id
    os.system("git checkout " + str(next_commit_id))
    os.chdir(os.path.normpath("../.."))


def get_top_k_wrong_files(right_files, br_corpus, java_files):
    randomly_sampled = random.sample(list(java_files), 100)

    all_files = []
    for filename in [f for f in randomly_sampled if f not in right_files]:
        try:
            raw_class_names = java_files[filename].split(" class ")[1:]

            class_names = []
            for block in raw_class_names:
                class_names.append(block.split(' ')[0])
            class_corpus = ' '.join(class_names)

            one = cosine_sim(br_corpus, java_files[filename])
            two = cosine_sim(br_corpus, class_corpus)

            file_info = [filename, one, two]
            all_files.append(file_info)
        except Exception:
            print("Error in wrong file parsing")
            del java_files[filename]

    topfifty = sorted(all_files, key=lambda x: x[1], reverse=True)[
        :len(right_files)]
    return topfifty


def calculate_scores(id, buggy_src_file, java_files, raw_corpus, date, bug_report, bug_reports, report_time, match):
    buggy_src_file = os.path.normpath(buggy_src_file)
    try:
        src = java_files[buggy_src_file]
    except:
        return

    rVSM_text_similarity = cosine_sim(raw_corpus, src)

    prev_reports = get_previous_report_by_filename(
        buggy_src_file, date, bug_reports)
    related_corpus = []
    for report in prev_reports:
        related_corpus.append(report["rawCorpus"])
    related_string = ' '.join(related_corpus)
    collaborative_filter_score = cosine_sim(raw_corpus, related_string)

    raw_class_names = src.split(" class ")[1:]

    class_names = []
    for block in raw_class_names:
        class_names.append(block.split(' ')[0])
    class_corpus = ' '.join(class_names)
    classname_similarity = cosine_sim(bug_report["rawCorpus"], class_corpus)

    mrReport = get_most_recent_report(
        buggy_src_file, convert_to_datetime(report_time), bug_reports)
    bug_fixing_recency_ = bug_fixing_recency(bug_report, mrReport)

    bug_fixing_frequency_ = bug_fixing_frequency(
        buggy_src_file, date, bug_reports)

    our_features_path = os.path.normpath('./data/our_features.csv')
    with open(our_features_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id, buggy_src_file, rVSM_text_similarity, collaborative_filter_score,
                         classname_similarity, bug_fixing_recency_, bug_fixing_frequency_, match])


stemmer = PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
words = stopwords.words("english")


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens if item not in words]


def normalize(text):
    return stem_tokens(word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(
    tokenizer=normalize, min_df=1, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


files = {}
start_dir = os.path.normpath("./data/eclipse.platform.ui")


def getAllCorpus():
    for dir, dir_names, file_names in os.walk(start_dir):
        for filename in [f for f in file_names if f.endswith(".java")]:
            src_name = os.path.join(dir, filename)
            src_file = open(src_name, 'r')
            src = src_file.read()
            src_file.close()

            file_key = src_name.split(start_dir)[1]
            file_key = file_key[len(os.sep):]
            files[file_key] = src

    return files


def get_previous_report_by_filename(filename, brdate, dictionary):
    return [br for br in dictionary if (filename in br["files"] and convert_to_datetime(br["report_time"]) < brdate)]


def convert_to_datetime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")


def get_months_between(d1, d2):
    date1 = convert_to_datetime(d1)
    date2 = convert_to_datetime(d2)
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)


def get_most_recent_report(filename, current_date, dictionary):
    matching_reports = get_previous_report_by_filename(
        filename, current_date, dictionary)
    if len(matching_reports) > 0:
        return max((br for br in matching_reports), key=lambda x: convert_to_datetime(x.get("report_time")))
    else:
        return None


def bug_fixing_recency(report1, report2):
    if report1 is None or report2 is None:
        return 0
    else:
        return 1/float(get_months_between(report1.get("report_time"), report2.get("report_time")) + 1)


def bug_fixing_frequency(filename, date, dictionary):
    return len(get_previous_report_by_filename(filename, date, dictionary))


def collaborative_filtering_score(report, filename, dictionary):
    matching_reports = get_previous_report_by_filename(
        filename, convert_to_datetime(report.get("report_time")), dictionary)
