# -------------------------------------------------------
# Assignment 2
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

# todo at first hard-coded classes, but decided to make it more flexible

# todo remove hard-coding of classes


import csv
import string
import time
import re
import sys
import pathlib
import itertools
import math
import numpy as np

""" Constants """
DELTA = 0.5
CLASSES = ('story', 'ask_hn', 'show_hn', 'poll')
MODEL_OUTPUT_FILE = "model-2018"
BASELINE_OUTPUT_FILE = "baseline-result"
DEFAULT_DATA_PATH = "hns_2018_2019.csv"


def main():

    if len(sys.argv) == 1:
        csv_file_path = DEFAULT_DATA_PATH
    elif len(sys.argv) == 2:
        csv_file_path = sys.argv[1]
    else:
        print("Usage: nb_classifier [file_path]")
        return

    ###########################
    # TASK 1
    # load data and train model
    ###########################

    training_set, testing_set = load_data(csv_file_path)

    feature_vectors, correct_labels, removed_words = prepare_data(training_set)

    start_time = time.time()

    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, DELTA)
    print(f'\nModel constructed in {time.time() - start_time} seconds.\n')

    print_basic_model_stats(nb_classifier)
    output_model_to_file(nb_classifier, removed_words)

    # todo check class_count_ and feature_count_ from sklearn
    # todo and generate model.txt and compare

    # Task 2
    # classify

    test_vectors, test_correct_labels, _ = prepare_data(testing_set)
    nb_scores, nb_labels, metrics = nb_classifier.test(test_vectors, test_correct_labels)
    print_metrics(metrics)
    output_test_results_to_file(nb_scores, nb_labels, test_correct_labels, testing_set, nb_classifier.classes, BASELINE_OUTPUT_FILE)


    # todo compare with same model with sklearn

    # Task3: implement way to pass a function to the trainer to filter words.
    # re-train, use same variables (dont hog memory)
    # add filters
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


def prepare_data(data_set):

    feature_vectors = []
    classifications = []
    removed_words = []

    # tokenize entries, count frequencies
    for row in data_set:

        class_name = row[3].strip()  # retrieve class name
        classifications.append(class_name)

        tokens = tokenize_row(row, removed_words)
        feature_vectors.append(tokens)

    return feature_vectors, classifications, removed_words


class NaiveBayesClassifier:

    def __init__(self, feature_vectors, correct_labels, delta):
        self.feature_vectors = feature_vectors
        self.correct_labels = correct_labels
        self.delta = delta

        # train

        self.classes = tuple({class_name for class_name in self.correct_labels})

        # initialize all mappings to zero
        self.model = {}
        self.class_frequencies = {class_name: 0 for class_name in self.classes}
        self.class_prior_likelihoods = {class_name: 0 for class_name in self.classes}
        self.word_frequencies = {class_name: 0 for class_name in self.classes}

        # add features to model and/or increment class frequencies for feature and overall;
        for (vector, class_name) in zip(self.feature_vectors, self.correct_labels):

            self.class_frequencies[class_name] += 1

            for feature in vector:
                if feature not in self.model:
                    self.model[feature] = FeatureProperties(self.classes)
                self.model[feature].frequencies[class_name] += 1
                self.word_frequencies[class_name] += 1

        # extract conditional probabilities
        vocabulary_size = len(self.model)

        for feature in self.model.keys():
            feature_properties = self.model[feature]
            for class_name in self.classes:
                # building P( w_i | c )
                cond_prob = feature_properties.frequencies[class_name] + self.delta
                cond_prob = cond_prob / (self.word_frequencies[class_name] + vocabulary_size * self.delta)
                feature_properties.likelihoods[class_name] = cond_prob

        # get class priors
        n_entries = len(correct_labels)
        for class_name in self.classes:
            self.class_prior_likelihoods[class_name] = (self.class_frequencies[class_name]) / n_entries

    def classify(self, vectors):
        nb_scores = []
        nb_classes = []
        for vector in vectors:
            scores = []
            for class_name in self.classes:
                score = math.log(self.class_prior_likelihoods[class_name], 10)
                for feature in vector:
                    if feature in self.model:
                        score += math.log(self.model[feature].likelihoods[class_name], 10)
                scores.append(score)
            nb_scores.append(scores)
            most_likely_class = self.classes[scores.index(max(scores))]
            nb_classes.append(most_likely_class)  # append most likely classification

        return nb_scores, nb_classes

    def test(self, test_vectors, test_correct_labels):
        nb_scores, nb_labels = self.classify(test_vectors)
        metrics = self.get_metrics(nb_labels, test_correct_labels)

        return nb_scores, nb_labels, metrics

    def get_metrics(self, nb_labels, correct_labels):

        conf_mat = np.zeros((len(self.classes), len(self.classes)), int)
        # to convert class_name to indices
        label_index = {class_name: index for (class_name, index) in zip(self.classes, range(len(self.classes)))}

        for (nb_label, correct_label) in zip(nb_labels, correct_labels):
            # get indices
            nb_label_i = label_index[nb_label]
            correct_label_i = label_index[correct_label]
            # populate confusion matrix
            conf_mat[nb_label_i][correct_label_i] += 1

        accuracy = sum(conf_mat.diagonal()) / conf_mat.sum()
        precision_vector = conf_mat.diagonal() / np.sum(conf_mat, 1)
        recall_vector = conf_mat.diagonal() / np.sum(conf_mat, 0)
        f1_vector = (2 * p * r / (p + r) for (p, r) in zip(precision_vector, recall_vector))

        precision = {class_name: value for (class_name, value) in zip(self.classes, precision_vector)}
        recall = {class_name: value for (class_name, value) in zip(self.classes, recall_vector)}
        f1 = {class_name: value for (class_name, value) in zip(self.classes, f1_vector)}

        return Metrics(accuracy, precision, recall, f1)

        # Below, without confusion matrix
        # start = time.time()
        # correct = {class_name: 0 for class_name in self.classes}
        # incorrect = {class_name: 0 for class_name in self.classes}
        # missing = {class_name: 0 for class_name in self.classes}
        #
        # for (nb_label, correct_label) in zip(nb_labels, correct_labels):
        #     if nb_label == correct_label:
        #         correct[nb_label] += 1
        #     else:
        #         incorrect[nb_label] += 1
        #         missing[correct_label] += 1
        #
        # # computing metrics
        # metrics = Metrics()
        # metrics.accuracy = sum(correct.values()) / len(nb_labels)
        # for class_name in self.classes:
        #     correct_label_count = correct[class_name]
        #     precision = correct_label_count / (correct_label_count + incorrect[class_name])
        #     recall = correct_label_count / (correct_label_count + missing[class_name])
        #     f1 = 2 * recall * precision / (recall + precision)
        #
        #     metrics.precision[class_name] = precision
        #     metrics.recall[class_name] = recall
        #     metrics.f1[class_name] = f1
        #
        # print(f'\nmetrics took {time.time() - start} seconds\n')
        # return metrics


class FeatureProperties:
    """
    This class is used to hold classification data for each unique word
    in a word dictionary
    """
    def __init__(self, classes):

        # frequencies
        self.frequencies = {class_name: 0 for class_name in classes}

        # conditional probabilities
        self.likelihoods = {class_name: 0 for class_name in classes}


class Metrics:

    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1


def tokenize_row(row, removed_words, *args):
    """
    Returns a list of tokens from an entry.
    It separates words separated by a forward slash,
    removes punctuation at beginning and end of a token,
    removes "'s" endings,
    remove tokens which consist of a lonely punctuation sign, contain a digit or are an empty string.
    :param row:
    :param removed_words:
    :param args: any number of additional filtering functions to apply to the tokenization process
    :return:
    """
    punctuation = string.punctuation
    punctuation += "“”‘’«"
    title = row[2].replace('/', ' ')
    tokens = title.split()
    tokens = [token.strip(punctuation).lower() for token in tokens if token not in punctuation]
    tokens = [re.sub(r'[\'’]s$', '', token) for token in tokens]  # removing "'s" endings
    # remove words with digits and empty strings
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if not token:
            del tokens[i]
            n -= 1
        elif re.match(r'.*\d.*', token):
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


def print_basic_model_stats(nb_classifier):
    print("BASIC MODEL STATS\n")
    print(f"size of vocabulary: {len(nb_classifier.model)}")
    print(f'classification counts: {nb_classifier.class_frequencies}')
    print(f'words per class: {nb_classifier.word_frequencies}')
    print(f'Priors: {nb_classifier.class_prior_likelihoods}')
    print()


def print_metrics(metrics):
    print("TEST METRICS\n")
    print("Accuracy: {0:.4f}".format(metrics.accuracy))
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.precision.items()]
    print(f"Precision: {''.join(line)}")
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.recall.items()]
    print(f"Recall: {''.join(line)}")
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.f1.items()]
    print(f"F1 Scores: {''.join(line)}")
    print()


# todo write class name in output
def output_model_to_file(classifier, removed_words):
    # prepare model output file
    model_lines = []
    line_counter = 1
    vocabulary = []

    for feature, feature_properties in sorted(classifier.model.items()):
        vocabulary.append(f'{feature}\n')

        line = [f'{line_counter}  {feature}']  # using a list to avoid multiple concatenations
        for class_name in classifier.classes:
            line.append(
                f'  {feature_properties.frequencies[class_name]}  '
                f'{"{0:.9f}".format(feature_properties.likelihoods[class_name])}')
        line.append('\n')

        model_lines.append(''.join(line))

        line_counter += 1

    model_file_name = f'{MODEL_OUTPUT_FILE}.txt'

    with open(model_file_name, "w", encoding='utf-8') as model_file, \
            open("vocabulary.txt", "w", encoding='utf-8') as voc_file, \
            open("removed_words.txt", "w", encoding='utf-8') as rem_file:
        model_file.writelines(model_lines)
        voc_file.writelines(vocabulary)
        rem_file.writelines(removed_words)


# todo add classname or otherwise show order
def output_test_results_to_file(nb_scores, nb_labels, test_correct_labels, testing_set, classes, file_name):
    lines = []
    line_counter = 1

    # outputting results to file
    for (nb_score_vector, nb_label, correct_label, row) in zip(nb_scores, nb_labels, test_correct_labels, testing_set):

        right_wrong = "right" if nb_label == correct_label else "wrong"

        line = [f'{line_counter}  {row[2]}  {nb_label}']  # using a list to avoid multiple concatenations
        for i in range(len(classes)):
            line.append(f'  {"{0:.6f}".format(nb_score_vector[i])}')
        line.append(f'  {correct_label}  {right_wrong}\n')
        lines.append(''.join(line))
        line_counter += 1

    file_name = f'{file_name}.txt'

    with open(file_name, "w", encoding='utf-8') as file:
        file.writelines(lines)


if __name__ == "__main__":
    main()

