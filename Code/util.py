import csv
import re
import os
import random
from datetime import datetime
from nltk.tokenize import  word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def tsv2dict():
    file_  = os.path.normpath('./data/Eclipse_Platform_UI.txt')
    reader = csv.DictReader(open(file_, 'r'), delimiter='\t')
    dict_list = []
    for line in reader:
        line["files"] = [s.strip() + ".java" for s in line["files"].split(".java") if s]
        line["rawCorpus"] = line["summary"][10:] + line["description"]
        line["summary"], line["description"], combinedCorpus = getCombinedCorpus(line)

        dict_list.append(line)
    return dict_list


def getCombinedCorpus(report):
    report["summary"] = cleanAndSplit(report["summary"])[2:]  
    report["description"] = cleanAndSplit(report["description"])
    combinedCorpus = report["summary"] + report["description"]
    return report["summary"], report["description"], combinedCorpus

def cleanAndSplit(text):
    
    returnText = re.sub(r'[^\w\s]','',text)

    returnText = [s.strip() for s in returnText.split()]
    return returnText


def csv2dict():
    file_ = os.path.normpath('./data/features.csv')
    with open(file_, 'r') as f:
        reader = csv.DictReader(f, delimiter=',') 
        csvDict = list()
        for line in reader:
            csvDict.append(line)   
                     
    return csvDict

def git_checkout(commmit_id):
    eclipse_ui_git_directory = os.path.normpath("./data/eclipse.platform.ui")
    os.chdir(eclipse_ui_git_directory)

    next_commit_id = commmit_id
    os.system("git checkout " + str(next_commit_id))
    os.chdir(os.path.normpath("../.."))

def get_top_k_wrong_files(rightFiles, brCorpus, javaFiles):
    randomlySampled = random.sample(list(javaFiles), 100)  

    allFiles = []
    for filename in [f for f in randomlySampled if f not in rightFiles]:
        try:
            rawClassNames = javaFiles[filename].split(" class ")[1:]

            classNames = []
            for block in rawClassNames:
                classNames.append(block.split(' ')[0])
            classCorpus = ' '.join(classNames)

            one = cosine_sim(brCorpus, javaFiles[filename])
            two = cosine_sim(brCorpus, classCorpus)

            fileInfo = [filename, one, two]
            allFiles.append(fileInfo)
        except Exception:
            print("Error in wrong file parsing")
            del javaFiles[filename]

    topfifty = sorted(allFiles, key=lambda x: x[1], reverse=True)[:len(rightFiles)]
    return topfifty

def calculate_scores(id, buggy_src_file, javaFiles, rawCorpus, date, bug_report, bug_reports, report_time, match):
    buggy_src_file = os.path.normpath(buggy_src_file)
    try:
        src = javaFiles[buggy_src_file]
    except:
        return

    rVSMTextSimilarity = cosine_sim(rawCorpus, src)

    prevReports = getPreviousReportByFilename(buggy_src_file, date, bug_reports)
    relatedCorpus = []
    for report in prevReports:
        relatedCorpus.append(report["rawCorpus"])
    relatedString = ' '.join(relatedCorpus)
    collaborativeFilterScore = cosine_sim(rawCorpus, relatedString)

    rawClassNames = src.split(" class ")[1:]

    classNames = []
    for block in rawClassNames:
        classNames.append(block.split(' ')[0])
    classCorpus = ' '.join(classNames)
    classNameSimilarity = cosine_sim(bug_report["rawCorpus"], classCorpus)

    mrReport = getMostRecentReport(buggy_src_file, convertToDateTime(report_time), bug_reports)
    bugFixingRecency_ = bugFixingRecency(bug_report, mrReport)

    bugFixingFrequency_ = bugFixingFrequency(buggy_src_file, date, bug_reports)

    our_features_path = os.path.normpath('./data/our_features.csv')
    with open(our_features_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([id, buggy_src_file, rVSMTextSimilarity, collaborativeFilterScore, classNameSimilarity, bugFixingRecency_, bugFixingFrequency_, match])

stemmer = PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
words = stopwords.words("english")
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens if item not in words]

def normalize(text):
    return stem_tokens(word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, min_df = 1, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

files = {}
start_dir = os.path.normpath("./data/eclipse.platform.ui")

def getAllCorpus():
    for dir,dirNames,fileNames in os.walk(start_dir):
        for filename in [f for f in fileNames if f.endswith(".java")]:
            srcName = os.path.join(dir, filename)
            srcFile = open(srcName, 'r')
            src = srcFile.read()
            srcFile.close()

            fileKey = srcName.split(start_dir)[1]
            fileKey = fileKey[len(os.sep):]
            files[fileKey] = src

    return files

def getPreviousReportByFilename(filename, brdate, dictionary):
    return [br for br in dictionary if (filename in br["files"] and convertToDateTime(br["report_time"]) < brdate)]

def convertToDateTime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

def getMonthsBetween(d1, d2):
    date1 = convertToDateTime(d1)
    date2 = convertToDateTime(d2)
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)

def getMostRecentReport(filename, currentDate, dictionary):
    matchingReports = getPreviousReportByFilename(filename, currentDate, dictionary)
    if len(matchingReports) > 0:
        return max((br for br in matchingReports), key=lambda x:convertToDateTime(x.get("report_time")))
    else:
        return None

def bugFixingRecency(report1, report2):
    if report1 is None or report2 is None:
        return 0
    else:
        return 1/float(getMonthsBetween(report1.get("report_time"), report2.get("report_time")) + 1)

def bugFixingFrequency(filename, date, dictionary):
    return len(getPreviousReportByFilename(filename, date, dictionary))

def collaborativeFilteringScore(report, filename, dictionary):
    matchingReports = getPreviousReportByFilename(filename, convertToDateTime(report.get("report_time")), dictionary)