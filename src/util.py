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


def git_clone(repo_url="https://github.com/eclipse/eclipse.platform.ui.git", clone_path="../data/"):
    """ Clones the git repo from repo_url

    Keyword Arguments:
        repo_url {str} -- [description] (default: {"https://github.com/eclipse/eclipse.platform.ui.git"})
        clone_path {str} -- [description] (default: {"../data/"})
    """
    repo_name = repo_url[repo_url.rfind('/')+1:-4]
    if os.path.isdir(clone_path + repo_name):
        print("Already cloned")
        return
    cwd = os.getcwd()
    if not os.path.isdir(clone_path):
        os.mkdir(clone_path)
    os.chdir(clone_path)
    os.system("git clone {}".format(repo_url))
    os.chdir(cwd)


def tsv2dict(tsv_path='../data/Eclipse_Platform_UI.txt'):
    """ Converts a tab separated file into a dict

    Keyword Arguments:
        tsv_path {string} -- path to tsv file (default: {'../data/Eclipse_Platform_UI.txt'})
    """
    reader = csv.DictReader(open(tsv_path, 'r'), delimiter='\t')
    dict_list = []
    for line in reader:
        line["files"] = [os.path.normpath(f[8:]) for f in line["files"].strip(
        ).split() if f.startswith("bundles/") and f.endswith(".java")]
        line["raw_text"] = line["summary"] + line["description"]
        line["summary"] = clean_and_split(line["summary"][11:])
        line["description"] = clean_and_split(line["description"])
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S")

        dict_list.append(line)
    return dict_list


def csv2dict(csv_path='../data/features.csv'):
    """Converts a comma separated file into a dict

    Keyword Arguments:
        csv_path {string} -- path to csv file (default: {'../data/features.csv'})
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict


def clean_and_split(text):
    """ Remove all punctuation and split text strings into lists of words

    Arguments:
        text {string} -- input text
    """
    table = str.maketrans(dict.fromkeys(string.punctuation))
    clean_text = text.translate(table)
    word_list = [s.strip() for s in clean_text.strip().split()]
    return word_list


def get_top_k_wrong_files(right_files, br_corpus, java_files, k=50):
    """ Top k wrong files

    Arguments:
        right_files {[type]} -- [description]
        br_corpus {[type]} -- [description]
        java_files {[type]} -- [description]

    Keyword Arguments:
        k {int} -- [description] (default: {50})
    """

    # Randomly sample 100 out of the 3,981 files so it will be quicker
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

    topfifty = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]
    return topfifty


def calculate_scores(id, buggy_file, java_files, raw_corpus, date, bug_report, bug_reports, report_time, match):
    buggy_file = os.path.normpath(buggy_file)
    try:
        src = java_files[buggy_file]
    except:
        return

    rVSM_text_similarity = cosine_sim(raw_corpus, src)

    prev_reports = get_previous_report_by_filename(
        buggy_file, date, bug_reports)
    related_corpus = []
    for report in prev_reports:
        related_corpus.append(report["raw_text"])
    related_string = ' '.join(related_corpus)
    collaborative_filter_score = cosine_sim(raw_corpus, related_string)

    raw_class_names = src.split(" class ")[1:]

    class_names = []
    for block in raw_class_names:
        class_names.append(block.split(' ')[0])
    class_corpus = ' '.join(class_names)
    classname_similarity = cosine_sim(bug_report["raw_text"], class_corpus)

    mrReport = get_most_recent_report(
        buggy_file, report_time, bug_reports)
    bug_fixing_recency_ = bug_fixing_recency(bug_report, mrReport)

    bug_fixing_frequency_ = bug_fixing_frequency(
        buggy_file, date, bug_reports)

    our_features_path = os.path.normpath('../data/our_features.csv')
    with open(our_features_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id, buggy_file, rVSM_text_similarity, collaborative_filter_score,
                         classname_similarity, bug_fixing_recency_, bug_fixing_frequency_, match])


def stem_tokens(tokens):
    """ Remove stopword and stem

    Arguments:
        tokens {token list} -- tokens to stem 
    """
    stemmer = PorterStemmer()
    removed_stopwords = [stemmer.stem(
        item) for item in tokens if item not in stopwords.words("english")]

    return removed_stopwords


def normalize(text):
    """ Lowercase, remove punctuation, tokenize and stem

    Arguments:
        text {string} -- A text to normalize
    """
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)
    return stemmed_tokens


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(
        tokenizer=normalize, min_df=1, stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = ((tfidf * tfidf.T).A)[0, 1]
    return sim


def get_all_source_code(start_dir="../data/eclipse.platform.ui/bundles/"):
    """ Creates corpus starting from 'start_dir'

    Keyword Arguments:
        start_dir {string} -- directory path to start (default: {"../data/eclipse.platform.ui/bundles/"})
    """
    files = {}
    start_dir = os.path.normpath(start_dir)
    for dir_, dir_names, file_names in os.walk(start_dir):
        for filename in [f for f in file_names if f.endswith(".java")]:
            src_name = os.path.join(dir_, filename)
            with open(src_name, 'r') as src_file:
                src = src_file.read()

            file_key = src_name.split(start_dir)[1]
            file_key = file_key[len(os.sep):]
            files[file_key] = src

    return files


def get_months_between(date1, date2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((date1.year - date2.year) *
                         12 + date1.month - date2.month)
    return diff_in_months


def get_most_recent_report(filename, current_date, dictionary):
    """ Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        filename {string} -- [description]
        current_date {datetime} -- [description]
        dictionary {dictionary} -- [description]
    """
    matching_reports = previous_reports(
        filename, current_date, dictionary)
    if len(matching_reports) > 0:
        return max((br for br in matching_reports), key=lambda x: x.get("report_time"))
    else:
        return None


def previous_reports(filename, brdate, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- shared source file name
        brdate {datetime} -- until date
        bug_reports {dictionary} -- bug report dictionary
    """
    return [br for br in bug_reports if (filename in br["files"] and br["report_time"] < brdate)]


def bug_fixing_recency(report1, report2):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        report1 {dictionary} -- current bug report
        report2 {dictionary} -- most recent bug report
    """
    if report1 is None or report2 is None:
        return 0
    else:
        return 1/float(get_months_between(report1.get("report_time"), report2.get("report_time")) + 1)


def bug_fixing_frequency(filename, date, dictionary):
    """ Calculates the Bug Fixing Frequency as defined by Lam et al.

    Arguments:
        filename {string} -- filename fixed by the bug report
        date {datetime} -- date of current bug retport
        dictionary {dictionary} -- dictionary of all bug reports
    """
    return len(previous_reports(filename, date, dictionary))


def collaborative_filtering_score(report, filename, dictionary):
    """ [summary]

    Arguments:
        report {dictionary} -- The bug report we're calculating metadata for
        filename {string} -- the filename we're checking previous bug reports for
        dictionary {dictionary} -- [description]
    """
    matching_reports = previous_reports(
        filename, report.get("report_time"), dictionary)
