# -------------------------------------------------------
# Assignment 2
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import csv
import string
import time
import re
import sys
import pathlib

""" Constants """
DELTA = 0.5
CLASSES = {'story', 'ask_hn', 'show_hn', 'poll'}
MODEL_OUTPUT_FILE = "model-2018"
DEFAULT_DATA_PATH = "hns_2018_2019.csv"

#todo add delta as method paramter, pass it the default


def main():

    if len(sys.argv) == 1:
        csv_file_path = DEFAULT_DATA_PATH
    elif len(sys.argv) == 2:
        csv_file_path = sys.argv[1]
    else:
        print("Usage: nb_classifier [file_path]")
        return

    # TASK 1
    # load data and train model
    training_set, testing_set = load_data(csv_file_path)

    classification_counts = {class_name: 0 for class_name in CLASSES}
    class_word_counts = {class_name: 0 for class_name in CLASSES}
    class_priors = {class_name: 0 for class_name in CLASSES}

    model, removed_words = train_model(training_set, class_word_counts, classification_counts, class_priors)
    output_model_to_file(model, removed_words)

    # todo check class_count_ and feature_count_ from sklearn
    # todo and generate model.txt and compare

    # todo remove below?
    print(f"size of vocabulary: {len(model)}")
    print(f'classification counts: {classification_counts}')
    print(f'words per class: {class_word_counts}')
    print(f'Priors: {class_priors}')

    # Task 2
    # classify
    # todo check make sure results same whether classify from model in memory or from file
    # todo compare with same model with sklearn
    # output to terminal perf measure accuracy, precision, recall fmeasure etc.

    # Task3: implement way to pass a function to the trainer to filter words.
    #todo for both procedures, plot a multi bar histogram:
    # for each x, plot a bar for each metric, maybe write f1-measure at the bottom
    # if time permits, test if rebuilding model adds to experiment


def load_data(csv_file_path):
    # keeps track of how many models have been generated, to keep distinct names and avoid over-writing files
    training_set = []
    testing_set = []

    # read csv entries and separate training and testing sets
    try:
        with open(csv_file_path, encoding="utf-8", newline='') as csv_file, \
                open("error_log.txt", "w", encoding="utf-8") as error_log:

            reader = csv.reader(csv_file)
            next(reader)  # skip header row
            for row in reader:
                if row[5].startswith("2018"):
                    training_set.append(row)
                elif row[5].startswith("2019"):
                    testing_set.append(row)
                else:
                    error_log.write(f"bad formatting: {row}\n")

    except OSError:
        print("\nFile cannot be found as specified.\n")
        exit()

    return training_set, testing_set


def train_model(training_set, class_wordcounts, classification_counts, class_priors):
    """
    Returns a trained model based on the training set
    :param training_set: a list of well-formatted rows
    :param class_wordcounts:
    :param classification_counts:
    :param class_priors:
    :return:
    """

    model = {}
    start = time.time()
    removed_words = []

    # tokenize entries, count frequencies
    for row in training_set:

        class_name = row[3].strip()  # retrieve class name
        classification_counts[class_name] += 1

        tokens = tokenize_row(row, removed_words)

        # increment class counts, add word to model
        for word in tokens:
            if word not in model:
                model[word] = WordEntry()
            model[word].frequencies[class_name] += 1
            class_wordcounts[class_name] += 1

    # extract conditional probabilities
    vocabulary_size = len(model)

    for word in model:
        entry = model[word]
        for class_name in CLASSES:
            # building P( w_i | c )
            cond_prob = entry.likelihoods[class_name] + DELTA
            cond_prob = cond_prob / (class_wordcounts[class_name] + vocabulary_size * DELTA)
            entry.likelihoods[class_name] = cond_prob

    print(f'\nModel constructed in {time.time() - start} seconds.\n')

    # get class priors
    n_entries = sum(classification_counts.values())
    for class_name in CLASSES:
        class_priors[class_name] = classification_counts[class_name] / n_entries

    return model, removed_words


def tokenize_row(row, removed_words, *args):
    """
    Returns a list of tokens from an entry.
    It separates words separated by a forward slash,
    removes punctuation at beginning and end of a token,
    removes "'s" endings,
    remove tokens which consist of a lonely punctuation sign, contain a digit or are an empty string.
    :param row:
    :return: a list of tokens
    """

    title = row[2].replace('/', ' ')
    tokens = title.split()
    tokens = [token.strip(string.punctuation).lower() for token in tokens if token not in string.punctuation]
    tokens = [re.sub(r'[\'’]s$', '', token) for token in tokens]  # removing "'s" endings
    # remove words with digits and empty strings
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if not token:
            del tokens[i]
            n -= 1
        if re.match(r'.*\d.*', token):
            removed_words.append(f'{token}\n')
            del tokens[i]
            n -= 1
        else:
            i += 1

    if len(args) > 0:
        for func in args:
            tokens = func(tokens)

    return tokens


def stop_word_filter(tokens, stop_list):
    pass


def word_length_filter(tokens):
    pass


def infrequent_word_filter(model):
    # remove don't rebuild for now
    # could use a stop list
    pass


def most_frequent_word_filter(model, percentage):
    # remove dont rebuild for now
    # could rebuild w a stop list
    pass


def output_model_to_file(model, removed_words):
    # prepare model output file
    model_lines = []
    line_counter = 1
    vocabulary = []

    for word, word_entry in sorted(model.items()):
        vocabulary.append(f'{word}\n')
        # creating the entry through a list
        # to avoid concatenation inefficiency
        # (since it is constructed through a loop)
        line = [f'{line_counter}  {word}']
        for class_name in CLASSES:
            line.append(
                f'  {word_entry.frequencies[class_name]}  {"{0:.9f}".format(word_entry.likelihoods[class_name])}')
        line.append('\n')

        model_lines.append(''.join(line))
        line_counter += 1

    model_file_name = f'{MODEL_OUTPUT_FILE}.txt'

    with open(model_file_name, "w", encoding='utf-8') as model_file, \
            open("vocabulary.txt", "w", encoding='utf-8') as voc_file, \
            open("removed_words.txt", "w", encoding='utf-8') as remov_file:
        model_file.writelines(model_lines)
        voc_file.writelines(vocabulary)
        remov_file.writelines(removed_words)


class WordEntry:
    """
    This class is used to hold classification data for each unique word
    in a word dictionary
    """
    def __init__(self):

        # frequencies
        self.frequencies = {class_name: 0 for class_name in CLASSES}

        # conditional probabilities
        self.likelihoods = {class_name: 0 for class_name in CLASSES}


if __name__ == "__main__":
    main()

