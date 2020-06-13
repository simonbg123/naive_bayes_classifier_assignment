# -------------------------------------------------------
# Assignment 2
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

import csv
import string
import re
import sys
import math
from matplotlib import pyplot as plt

""" Constants """
DELTA = 0.5
DEFAULT_DATA_PATH = "hns_2018_2019.csv"


def main():

    if len(sys.argv) == 1:
        csv_file_path = DEFAULT_DATA_PATH
    elif len(sys.argv) == 2:
        csv_file_path = sys.argv[1]
    else:
        print("Usage: nb_classifier [file_path]")
        return

    # TASK 1
    # load data
    training_set, testing_set = load_data(csv_file_path)

    # train model
    feature_vectors, correct_labels, removed_words = prepare_data(training_set)
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, DELTA)

    # print model stats and output model to file
    print("\n## BASELINE MODEL ##\n")
    model_output_file = "model-2018"
    baseline_output_file = "baseline-result"
    plot_title = "BASELINE MODEL"

    print_basic_model_stats(nb_classifier)
    output_model_to_file(nb_classifier, model_output_file)

    # output vocabulary, removed-words files
    vocabulary = nb_classifier.get_vocabulary()
    with open("vocabulary.txt", "w", encoding='utf-8') as voc_file, \
            open("removed_words.txt", "w", encoding='utf-8') as rem_file:
        voc_file.write('\n'.join(vocabulary))
        rem_file.write('\n'.join(removed_words))

    # TASK 2
    # prepare testing data - will also be used for experiments
    test_vectors, test_correct_labels, _ = prepare_data(testing_set)

    testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, baseline_output_file, plot_title)

    # TASK 3
    # Experiments with model manipulations

    # Experiment 1: stop-word filtering
    input("PRESS ENTER to continue with stop-word experiment\n")
    print("\n## STOP-WORD MODEL ##\n")

    model_output_file = "stopword-model"
    result_output_file = "stopword-result"
    plot_title = "STOP-WORD TEST"

    with open("stopwords.txt", "r", encoding='utf-8') as sw_file:
        stop_words = sw_file.read().split('\n')

    # make a stop-word filter to use in preparation of the data
    word_filter = word_filter_factory(stop_words)

    nb_classifier = training_routine(training_set, model_output_file, DELTA, word_filter)
    testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, result_output_file, plot_title)

    # Experiment 2: word lengths
    input("PRESS ENTER to continue with word-length experiment\n")
    print("\n## WORD-LENGTH MODEL ##\n")

    model_output_file = "wordlength-model"
    result_output_file = "wordlength-result"
    plot_title = "WORD-LENGTH TEST"

    nb_classifier = training_routine(training_set, model_output_file, DELTA, word_length_filter)
    testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, result_output_file,plot_title)

    # Experiment 3: frequency experiments
    input("PRESS ENTER to continue with frequency experiments\n")

    # Remove infrequent words experiment

    # get a fresh model
    feature_vectors, correct_labels, _ = prepare_data(training_set, word_length_filter)
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, DELTA)
    initial_word_count = len(nb_classifier.model)

    # conduct the experiment
    freq_list = (1, 5, 10, 15, 20)  # frequencies for which we remove words
    data_rem_infreq, x_ticks_infreq = infrequent_words_experiment(freq_list, nb_classifier, test_vectors, test_correct_labels)

    # Remove most frequent words experiment

    # get a brand new model
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, DELTA)

    # conduct experiment
    freq_list = (0.05, 0.1, 0.15, 0.2, 0.25)  # frequencies we will remove
    data_rem_freq, x_ticks_freq = frequent_words_experiment(freq_list, nb_classifier, test_vectors, test_correct_labels)

    # plot joint results
    plt.close()
    title = f"Frequency experiments\nInitial vocabulary {initial_word_count}"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    bar_plot(fig, ax[0], data_rem_infreq, x_ticks_infreq, "Removing Infrequent Words")
    bar_plot(fig, ax[1], data_rem_freq, x_ticks_freq, "Removing Most Frequent Words")
    fig.suptitle(title)
    plt.subplots_adjust(left=0.18)
    plt.show(block=False)

    input("\nPress Enter to quit")

    print("\nBye!\n")


def load_data(csv_file_path):

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


def training_routine(training_set, output_file, delta, word_filter=None):
    # train model
    feature_vectors, correct_labels, removed_words = prepare_data(training_set, word_filter)
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, delta)

    # print model stats and output model to file
    print_basic_model_stats(nb_classifier)
    output_model_to_file(nb_classifier, output_file)

    return nb_classifier


def testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, output_file, t):
    # test model, print stats and output results to file
    nb_scores, nb_labels, metrics = nb_classifier.test(test_vectors, test_correct_labels)
    print_metrics(metrics)
    output_test_results_to_file(nb_scores, nb_labels, test_correct_labels, testing_set, nb_classifier.classes, output_file)

    # plot results
    plt.close()
    x_ticks = ['accuracy']
    for class_name in nb_classifier.classes:
        x_ticks.append(f"F1 score: {class_name}")

    data = [metrics.accuracy]
    for class_name in nb_classifier.classes:
        data.append(metrics.f1[class_name])

    fig, ax = plt.subplots(figsize=(7, 5))

    plt.bar(x_ticks, data, width=0.7)
    for i, v in enumerate(data):
        ax.text(i - 0.15, v + 0.01, "{:.4f}".format(v), fontweight='bold')

    ax.set_title(t)
    plt.show(block=False)


def prepare_data(data_set, filter_func=None):

    feature_vectors = []
    classifications = []
    removed_words = []

    # tokenize entries, count frequencies
    for row in data_set:

        class_name = row[3].strip()  # retrieve class name
        classifications.append(class_name)

        tokens = tokenize_row(row, removed_words, filter_func)
        feature_vectors.append(tokens)

    return feature_vectors, classifications, removed_words


def tokenize_row(row, removed_words, filter_func=None):
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
            removed_words.append(token)
            del tokens[i]
            n -= 1
        else:
            i += 1

    if filter_func:
        tokens = filter_func(tokens, removed_words)

    return tokens


def word_filter_factory(stop_words):

    def func(tokens, removed_words):
        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            if token in stop_words:
                removed_words.append(token)
                del tokens[i]
                n -= 1
            else:
                i += 1

        return tokens

    return func


def word_length_filter(tokens, removed_words):
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if not 2 < len(token) < 9:
            removed_words.append(token)
            del tokens[i]
            n -= 1
        else:
            i += 1

    return tokens


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

        correct = {class_name: 0 for class_name in self.classes}
        incorrect = {class_name: 0 for class_name in self.classes}
        missing = {class_name: 0 for class_name in self.classes}

        for (nb_label, correct_label) in zip(nb_labels, correct_labels):
            if nb_label == correct_label:
                correct[nb_label] += 1
            else:
                incorrect[nb_label] += 1
                missing[correct_label] += 1

        # computing metrics
        accuracy = sum(correct.values()) / len(nb_labels)
        precision = {}
        recall = {}
        f1 = {}

        for class_name in self.classes:

            correct_label_count = correct[class_name]

            if incorrect[class_name] == 0 and missing[class_name] == 0:
                p = 1
                r = 1
                f = 1
            elif correct_label_count == 0:
                p = 0
                r = 0
                f = 0
            else:
                p = correct_label_count / (correct_label_count + incorrect[class_name])
                r = correct_label_count / (correct_label_count + missing[class_name])
                f = 2 * p * r / (p + r)

            precision[class_name] = p
            recall[class_name] = r
            f1[class_name] = f

        return Metrics(accuracy, precision, recall, f1)

    def get_vocabulary(self):
        return [word for word in sorted(self.model.keys())]


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


def print_basic_model_stats(nb_classifier):
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
def output_model_to_file(classifier, file_name):
    # prepare model output file
    model_lines = []
    line_counter = 1

    for feature, feature_properties in sorted(classifier.model.items()):

        line = [f'{line_counter}  {feature}']  # using a list to avoid multiple concatenations
        for class_name in classifier.classes:
            line.append(
                f'  {feature_properties.frequencies[class_name]}  '
                f'{"{0:.9f}".format(feature_properties.likelihoods[class_name])}')
        line.append('\n')

        model_lines.append(''.join(line))

        line_counter += 1

    model_file_name = f'{file_name}.txt'

    with open(model_file_name, "w", encoding='utf-8') as model_file:
        model_file.writelines(model_lines)


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


def infrequent_words_experiment(freq_list, nb_classifier, test_vectors, test_correct_labels):

    data_rem_infreq = {'accuracy': []}  # to stock metrics for each round of removal
    for class_name in nb_classifier.classes:
        key = f'F1 score: {class_name}'
        data_rem_infreq[key] = []

    x_ticks_infreq = []
    # getting metrics after each round of removal
    for frequency in freq_list:
        remove = []
        for word, properties in nb_classifier.model.items():
            if sum(properties.frequencies.values()) <= frequency:
                remove.append(word)

        for key in remove:
            del nb_classifier.model[key]

        _, _, metrics = nb_classifier.test(test_vectors, test_correct_labels)

        data = [metrics.accuracy]
        for val in metrics.f1.values():
            data.append(val)

        for key, val in zip(data_rem_infreq.keys(), data):
            data_rem_infreq[key].append(val)

        x_ticks_infreq.append(len(nb_classifier.model))

    return data_rem_infreq, x_ticks_infreq


def frequent_words_experiment(freq_list, nb_classifier, test_vectors, test_correct_labels):

    data_rem_freq = {'accuracy': []}  # to stock metrics for each round of removal
    for class_name in nb_classifier.classes:
        key = f'F1 score: {class_name}'
        data_rem_freq[key] = []

    # order keys in terms of frequency
    sorted_freq_keys = sorted(nb_classifier.model.keys(),
                              key=lambda k: sum(nb_classifier.model[k].frequencies.values()),
                              reverse=True)

    x_ticks_freq = []
    for frequency in freq_list:
        # get most frequent keys
        for i in range(math.floor(frequency * len(sorted_freq_keys))):
            if sorted_freq_keys[i] in nb_classifier.model:  # we may have already removed this key in previous round
                del nb_classifier.model[sorted_freq_keys[i]]

        _, _, metrics = nb_classifier.test(test_vectors, test_correct_labels)

        data = [metrics.accuracy]
        for val in metrics.f1.values():
            data.append(val)

        for key, val in zip(data_rem_freq.keys(), data):
            data_rem_freq[key].append(val)

        x_ticks_freq.append(len(nb_classifier.model))

    return data_rem_freq, x_ticks_freq


def bar_plot(fig, ax,  data, x_ticks, title, legend=True):

    total_width = 0.8
    single_width = 0.9

    # todo conf figure size

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    if legend:
        fig.legend(bars, data.keys(), loc='center left')

    x_ticks.insert(0, None)
    ax.set_xticklabels(x_ticks)
    ax.set_title(title)
    ax.set_xlabel('words remaining')


if __name__ == "__main__":
    main()

