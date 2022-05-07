from os import listdir

import preprocessing.preprocessing as prep

ROOT_PATH = prep.root_path
TRANSCRIPTION_PATH = prep.transcription_path
KEYWORD_PATH = ROOT_PATH / "task" / "keywords.txt"
CLASSIFICATION_PATH = ROOT_PATH / "distance"
IMG_PATH = prep.img_root


def read_classifications(path):
    classifications = {}
    for line in open(path, "r"):
        keyword, assigned_words = line.split(",", 1)
        assigned_words = assigned_words.split(",")
        idx = 0
        while idx < len(assigned_words):
            test_word_id = assigned_words[idx]
            dissimilarity = assigned_words[idx + 1]
            idx += 2
            classifications[test_word_id] = keyword
    return classifications


def read_keywords(path):
    keywords = []
    for keyword in open(path, "r"):
        keywords.append(keyword)
    return keywords


def read_transcriptions(path):
    transcriptions = {}
    for line in open(path, "r"):
        word_id, transcription = line.split()
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


def calculate_precision_and_recall(thresholds):
    keywords = read_keywords(KEYWORD_PATH)
    transcriptions = read_transcriptions(TRANSCRIPTION_PATH)
    classifications = read_classifications(CLASSIFICATION_PATH)
    images = [img for img in listdir(IMG_PATH)]

    for image in images:
        precision_scores = []
        recall_scores = []
        for k in thresholds:
            img_id = image.split(".")[0]

            classifications_per_img = {word_id: classification for (word_id, classification) in classifications.items()
                                       if word_id.startswith(img_id)}
            transcriptions_per_img = {word_id: transcription for (word_id, transcription) in transcriptions.items()
                                      if word_id.startswith(img_id)}

            true_positives, false_positives, false_negatives = \
                classification_values(classifications_per_img, transcriptions_per_img, keywords)

            precision_scores.append(true_positives / (true_positives + false_positives))
            recall_scores.append(true_positives / (true_positives + false_negatives))

        # TODO: precision-recall-curve per image
