import sys
import matplotlib.pyplot as plt
import time
from fpdf import FPDF
from pathlib import Path
sys.path.append("..")
from string_utils import correct_string

ROOT_PATH = Path.cwd().parents[0]
TRANSCRIPTION_PATH = ROOT_PATH / "data" / "ground-truth" / "transcription.txt"
KEYWORD_PATH = ROOT_PATH / "data" / "task" / "keywords.txt"
CLASSIFICATION_PATH = ROOT_PATH / "distance" / "output" / "distances.csv"
THRESHOLD = 1000


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


def top_k_matches(k, classifications, transcriptions, threshold):
    matches = []
    no_matches = []
    transcription_words = list(transcriptions.values())
    for word in classifications.keys():

        # get ids from transcriptions associated to this word
        ids = [word_id for word_id in transcriptions.keys() if transcriptions[word_id] == word]

        # get ids from classifications associated to this word (up to a specified threshold)
        classified_ids = [y[0] for y in classifications[word][:threshold+1]]
        if word not in transcription_words:
            print(f'No transcription for word: {word}')
            continue
        elif set(ids).isdisjoint(classified_ids):
            print(f'None of the classified ids found among the associated ids in the transcriptions for \'{word}\'')
            no_matches.append(word)
            continue

        # create match-list indicating whether a classified id corresponds to the word (1) or not (0),
        # i.e checking if id is among ids for this word according to the transcriptions
        idx = 0
        current_matches = []
        while sum([match[0] for match in current_matches]) < k and idx < threshold:
            current_matches.append(
                (int(transcriptions[classifications[word][idx][0]] == word), classifications[word][idx][1])
            )
            idx += 1

        # append matches for this word to overall matches
        matches.extend(current_matches)

    # sort matches by dissimilarity in ascending order
    matches.sort(key=lambda match: match[1])
    return [match[0] for match in matches], no_matches


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
    plt.legend(loc='upper right')
    plt.savefig('./precision_recall_curve.png')


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


def evaluation_report(no_matches, avg_precision, time_elapsed):

    # initialize new doc with empty page
    pdf = FPDF()
    pdf.add_page()

    # add title
    pdf.set_xy(0.0, 0.0)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(w=210.0, h=40.0, align='C', txt="Evaluation", border=0)

    # add list words with no match
    text = f'For the following words no match was found, i.e none of the classified ids for this word was found ' \
           f'among the ids associated to this word in the transcriptions for a pre-specified threshold ({THRESHOLD}):\n'
    pdf.set_xy(13.0, 30.0)
    pdf.set_font('Arial', 'B', 8)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(w=180.0, h=4.0, align='l', txt=text, border=0)

    no_matches = ''.join([word + ', ' for word in no_matches])
    pdf.set_xy(13.0, 40.0)
    pdf.set_font('Arial', '', 8)
    pdf.multi_cell(w=180.0, h=4.0, align='L', txt=no_matches, border=0)

    # add plot
    pdf.set_xy(13.0, 130.0)
    pdf.set_font('Arial', 'B', 8)
    pdf.multi_cell(w=180.0, h=4.0, align='L', txt='PRECISION-RECALL-CURVE', border=0)
    pdf.set_xy(13.0, 134.0)
    pdf.image('./precision_recall_curve.png', link='', type='', w=130, h=130)

    # avg precision and time elapsed
    avg_precision = "Average precision calculated: " + str(avg_precision)
    pdf.set_xy(13.0, 267.0)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(w=180.0, h=4.0, align='L', txt=avg_precision, border=0)

    time_elapsed = "Time for evaluation: " + str(time_elapsed) + " seconds"
    pdf.set_xy(13.0, 271.0)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(w=180.0, h=4.0, align='L', txt=time_elapsed, border=0)

    pdf.output('./evaluation_report.pdf', 'F')


def run_evaluation():
    start = time.time()
    classifications = read_classifications(CLASSIFICATION_PATH)
    transcriptions = read_transcriptions(TRANSCRIPTION_PATH)

    matches, no_matches = top_k_matches(5, classifications, transcriptions, THRESHOLD)

    precision, recall = precision_recall(matches)

    plot_precision_recall_curve(precision, recall)
    average_precision = calculate_average_precision(precision, recall)
    print("Average precision: " + str(average_precision))

    end = time.time()
    print("Time for evaluation: " + str(end - start) + " seconds")

    evaluation_report(no_matches, average_precision, end - start)


if __name__ == "__main__":
    run_evaluation()
