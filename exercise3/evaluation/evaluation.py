from os import listdir

import preprocessing.preprocessing as prep
import matplotlib.pyplot as plt

ROOT_PATH = prep.root_path
TRANSCRIPTION_PATH = prep.transcription_path
KEYWORD_PATH = ROOT_PATH / "task" / "keywords.txt"
CLASSIFICATION_PATH = ROOT_PATH / "distance"
IMG_PATH = prep.img_root
THRESHOLDS = range(5, 15)


def read_classifications(path, threshold):
    classifications = {}
    for line in open(path, "r"):
        keyword, assigned_words = line.split(",", 1)
        assigned_words = assigned_words.split(",")
        idx = 0
        number_of_words = 0
        while number_of_words < threshold:
            test_word_id = assigned_words[idx]
            dissimilarity = assigned_words[idx + 1]
            number_of_words += 1
            idx += 2
            classifications[test_word_id] = keyword
    return classifications


def read_keywords(path):
    keywords = []
    for keyword in open(path, "r"):
        keywords.append(keyword)
    return keywords


def read_transcriptions(path, classifications):
    transcriptions = {}
    for line in open(path, "r"):
        word_id, transcription = line.split()
        if word_id in classifications.keys():
            transcriptions[word_id] = transcription
    return transcriptions


def classification_values(classifications, transcriptions, keywords):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for word in keywords:
        correct_ids = [word_id for word_id, keyword in transcriptions.items() if keyword == word]
        classified_ids = [word_id for word_id, keyword in classifications.items() if keyword == word]
        true_positives += sum(list(map(lambda word_id: 1 if word_id in correct_ids else 0, classified_ids)))
        false_positives += sum(list(map(lambda word_id: 1 if word_id not in correct_ids else 0, classified_ids)))
        false_negatives += sum(list(map(lambda word_id: 1 if word_id not in classified_ids else 0, correct_ids)))
    return true_positives, false_positives, false_negatives


def plot_precision_recall_curve(precision_scores, recall_scores):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(recall_scores, precision_scores, label='DTW')
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.legend(loc='center left')


def calculate_average_precision(precision_scores, recall_scores):
    if len(precision_scores) != len(recall_scores):
        raise Exception("precision_scores and recall_scores must be of same length!")
    average_precision = 0
    idx = 1
    while idx < len(precision_scores):
        average_precision += (recall_scores[idx] - recall_scores[idx-1]) * precision_scores[idx]
    return average_precision


def calculate_precision_and_recall():
    keywords = read_keywords(KEYWORD_PATH)

    precision_scores = []
    recall_scores = []
    for threshold in THRESHOLDS:
        classifications = read_classifications(CLASSIFICATION_PATH, threshold)
        transcriptions = read_transcriptions(TRANSCRIPTION_PATH, classifications)

        true_positives, false_positives, false_negatives = \
            classification_values(classifications, transcriptions, keywords)

        try:
            precision = true_positives / (true_positives + false_positives)
        except:
            precision = 1

        try:
            recall = true_positives / (true_positives + false_negatives)
        except:
            recall = 1

        precision_scores.append(precision)
        recall_scores.append(recall)

    plot_precision_recall_curve(precision_scores, recall_scores)
    average_precision = calculate_average_precision(precision_scores, recall_scores)
    print("Average precision: " + str(average_precision))
