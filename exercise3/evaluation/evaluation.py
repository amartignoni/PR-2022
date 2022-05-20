import matplotlib.pyplot as plt
from pathlib import Path
from exercise3.string_utils import correct_string

ROOT_PATH = Path.cwd().parents[0]
TRANSCRIPTION_PATH = ROOT_PATH / "data" / "ground-truth" / "transcription.txt"
KEYWORD_PATH = ROOT_PATH / "data" / "task" / "keywords.txt"
CLASSIFICATION_PATH = ROOT_PATH / "distance" / "output" / "distances.csv"
THRESHOLDS = range(5, 6)


def read_classifications(path):
    classifications = {}
    for line in open(path, "r"):
        keyword, assigned_words = line.split(",", 1)
        assigned_words = assigned_words.split(",")
        word_id_dissimilarity_tuples = []
        idx = 0
        while idx < len(assigned_words):
            test_word_id = assigned_words[idx]
            dissimilarity = assigned_words[idx + 1]
            word_id_dissimilarity_tuples.append((test_word_id, dissimilarity))
            idx += 2
        classifications[keyword] = word_id_dissimilarity_tuples
    return classifications


def read_keywords(path):
    keywords = []
    for keyword in open(path, "r"):
        keywords.append(keyword.strip())
    return keywords


def read_transcriptions(path):
    transcriptions = {}
    for line in open(path, "r"):
        word_id, transcription = line.split()
        transcriptions[word_id] = correct_string(transcription)
    return transcriptions


def classification_values(classifications, transcriptions, keywords):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    classified_words = set(classifications.values())
    for word in classified_words:
        correct_ids = [word_id for word_id, keyword in transcriptions.items() if keyword == word.replace("-", "")]
        classified_ids = [
            word_id for word_id, keyword in classifications.items() if keyword == word
        ]
        true_positives += sum(
            list(
                map(lambda word_id: 1 if word_id in correct_ids else 0, classified_ids)
            )
        )
        false_positives += sum(
            list(
                map(
                    lambda word_id: 1 if word_id not in correct_ids else 0,
                    classified_ids
                )
            )
        )
        false_negatives += sum(
            list(
                map(
                    lambda word_id: 1 if word_id not in classified_ids else 0,
                    correct_ids
                )
            )
        )
    return true_positives, false_positives, false_negatives


def top_k_matches(k, classifications, transcriptions):
    matches = []
    transcription_words = list(transcriptions.values())
    for word in classifications.keys():
        ids = [word_id for word_id in transcriptions.keys() if transcriptions[word_id] == word]
        if word not in transcription_words:
            print(f'No transcription for word: {word}')
            continue
        elif set(ids).isdisjoint([y[0] for y in classifications[word]]):
            print(f'None of the classified word_ids for word \'{word}\' '
                  f'was found in associated ids in the transcriptions')
            continue
        idx = 0
        current_matches = []
        while sum([match[0] for match in current_matches]) < k and idx < 2433:
            current_matches.append(
                (int(transcriptions[classifications[word][idx][0]] == word), classifications[word][idx][1])
            )
            idx += 1
        matches.extend(current_matches)
    matches.sort(key=lambda match: match[1])
    return [match[0] for match in matches]


def precision_recall(matches):
    precision = []
    recall = []
    for index, match in enumerate(matches):
        print(index)
        precision.append((sum(matches[:index+1]) / (len(matches[:index+1]))))
        recall.append((sum(matches[:index+1]) / sum(matches)))
    return precision, recall


def plot_precision_recall_curve(precision_scores, recall_scores):
    plt.figure(figsize=(10, 10))
    plt.plot(recall_scores, precision_scores, label='DTW')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='center left')
    plt.savefig('./precision_recall_curve.png')
    # figure, axis = plt.subplots(figsize=(6, 6))
    # axis.plot(recall_scores, precision_scores, label="DTW")
    # axis.set_xlabel("Recall")
    # axis.set_ylabel("Precision")
    # axis.legend(loc="center left")


def calculate_average_precision(precision_scores, recall_scores):
    if len(precision_scores) != len(recall_scores):
        raise Exception("precision_scores and recall_scores must be of same length!")
    average_precision = 0
    idx = 1
    while idx < len(precision_scores):
        average_precision += (
                                     recall_scores[idx] - recall_scores[idx - 1]
                             ) * precision_scores[idx]
        idx += 1
    return average_precision


def run_evaluation():
    classifications = read_classifications(CLASSIFICATION_PATH)
    transcriptions = read_transcriptions(TRANSCRIPTION_PATH)

    matches = top_k_matches(5, classifications, transcriptions)

    precision, recall = precision_recall(matches)

    plot_precision_recall_curve(precision, recall)
    average_precision = calculate_average_precision(precision, recall)
    print("Average precision: " + str(average_precision))


if __name__ == "__main__":
    run_evaluation()
