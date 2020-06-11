# -------------------------------------------------------
# Assignment 2
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

# todo at first hard-coded classes, but decided to make it more flexible

import csv
import string
import time
import re
import sys
import pathlib
import itertools
import math

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

    feature_vectors, classifications, removed_words = prepare_data(training_set)

    start_time = time.time()

    nb_classifier = NaiveBayesClassifier(feature_vectors, classifications, DELTA, CLASSES)
    print(f'\nModel constructed in {time.time() - start_time} seconds.\n')

    output_model_to_file(nb_classifier, removed_words)

    print("BASIC MODEL STATS\n")
    print(f"size of vocabulary: {len(nb_classifier.model)}")
    print(f'classification counts: {nb_classifier.class_frequencies}')
    print(f'words per class: {nb_classifier.word_frequencies}')
    print(f'Priors: {nb_classifier.class_prior_likelihoods}')

    # todo check class_count_ and feature_count_ from sklearn
    # todo and generate model.txt and compare

    # Task 2
    # classify

    testing_vectors, testing_classifications, _ = prepare_data(testing_set)
    test_classification = nb_classifier.classify(testing_vectors)

    # todo one method to output to file, one method to get metrics.
    metrics = process_results(test_classification, testing_set, BASELINE_OUTPUT_FILE, nb_classifier)

    print("\nTEST METRICS\n")
    print("Accuracy: {0:.4f}".format(metrics.accuracy))
    line = ["'{}': {:.4f}".format(class_name, value) for class_name, value in metrics.precision.items()]
    print(f"Precision: {''.join(line)}")
    line = ["'{}': {:.4f}".format(class_name, value) for class_name, value in metrics.recall.items()]
    print(f"Recall: {''.join(line)}")
    line = ["'{}': {:.4f}".format(class_name, value) for class_name, value in metrics.f1.items()]
    print(f"F1 Scores: {''.join(line)}")

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

    def __init__(self, feature_vectors, classifications, delta, classes=None):
        self.feature_vectors = feature_vectors
        self.classifications = classifications
        self.delta = delta

        # train

        if classes:
            self.classes = classes
        else:   # retrieve class names
            self.classes = {class_name for class_name in self.classifications}
            #todo make sure order is stable

        # initialize all mappings to zero
        self.model = {}
        self.class_frequencies = {class_name: 0 for class_name in self.classes}
        self.class_prior_likelihoods = {class_name: 0 for class_name in self.classes}
        self.word_frequencies = {class_name: 0 for class_name in self.classes}

        # add features to model and/or increment class frequencies for feature and overall;
        for (vector, class_name) in zip(self.feature_vectors, self.classifications):

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
        n_entries = sum(self.class_frequencies.values())
        # smoothing becomes necessary since the assignment requires probabilities for poll class
        # but poll class in absent from the training set
        for class_name in self.classes:
            self.class_prior_likelihoods[class_name] = (self.class_frequencies[class_name] + self.delta) / n_entries

    def classify(self, vectors):
        nb_scores = []
        for vector in vectors:
            scores = []
            for class_name in self.classes:
                score = math.log(self.class_prior_likelihoods[class_name], 10)
                for feature in vector:
                    if feature in self.model:
                        score += math.log(self.model[feature].likelihoods[class_name], 10)
                scores.append(score)
            scores.append(self.classes[scores.index(max(scores))])  # append most likely classification
            nb_scores.append(scores)

        return nb_scores


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

    def __init__(self):
        self.accuracy = 0
        self.precision = {}
        self.recall = {}
        self.f1 = {}


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


def process_results(results, testing_set, file_name, model):
    lines = []
    line_counter = 1
    correct_labels = {class_name : 0 for class_name in model.classes}
    incorrect_labels = {class_name : 0 for class_name in model.classes}
    missing_labels = {class_name : 0 for class_name in model.classes}

    # outputting results to file and compiling data for metrics
    for (result, row) in zip(results, testing_set):

        model_label = result[-1]
        right_label = row[3]

        if model_label == right_label:
            right_wrong = "right"
            correct_labels[model_label] += 1
        else:
            right_wrong = "wrong"
            incorrect_labels[model_label] += 1
            missing_labels[right_label] += 1

        line = [f'{line_counter}  {row[2]}  {result[-1]}']  # using a list to avoid multiple concatenations
        for i in range(len(model.classes)):
            line.append(f'  {"{0:.6f}".format(result[i])}')
        line.append(f'  {row[3]}  {right_wrong}\n')
        line_counter += 1
        lines.append(''.join(line))

    # computing metrics
    metrics = Metrics()
    metrics.accuracy = sum(correct_labels.values()) / len(results)
    for class_name in model.classes:
        correct_label_count = correct_labels[class_name]

        if correct_label_count == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = correct_label_count / (correct_label_count + incorrect_labels[class_name])
            recall = correct_label_count / (correct_label_count + missing_labels[class_name])
            f1 = 2 * recall * precision / (recall + precision)

        metrics.precision[class_name] = precision
        metrics.recall[class_name] = recall
        metrics.f1[class_name] = f1

    file_name = f'{file_name}.txt'

    with open(file_name, "w", encoding='utf-8') as file:
        file.writelines(lines)

    return metrics


if __name__ == "__main__":
    main()

