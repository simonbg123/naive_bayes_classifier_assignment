# -------------------------------------------------------
# Assignment 2
# Written by Simon Brillant-Giroux, 40089110
# For COMP 472 Section ABIX – Summer 2020
# --------------------------------------------------------

# todo read sign expectations of originality
# todo finish documentation and README file
# todo show values in bar charts

import csv
import string
import re
import sys
import time
from matplotlib import pyplot as plt
from naive_bayes_classifier import *

""" Constants """
DELTA = 0.5
DEFAULT_DATA_PATH = "hns_2018_2019.csv"


def main():
    """

    :return:
    """

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
    feature_vectors, correct_labels, removed_words = prepare_data(training_set, None, True)
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

    if not testing_set:
        print("\nNo testing set was supplied. Program will exit.\nBye!\n")
        return

    testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, baseline_output_file, plot_title)

    """
    TASK 3
    Experiments with model manipulations
    """

    """Experiment 1: stop-word filtering"""
    input("PRESS ENTER to continue with stop-word experiment\n")
    print("\n## STOP-WORD MODEL ##\n")

    model_output_file = "stopword-model"
    result_output_file = "stopword-result"
    plot_title = "STOP-WORD TEST"

    with open("stopwords.txt", "r", encoding='utf-8') as sw_file:
        stop_words = sw_file.read().split('\n')

    stop_words = {word for word in stop_words}

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

    """Experiment 3: frequency experiments"""
    input("PRESS ENTER to continue with frequency experiments\n")
    print("Please wait: removing words and retraining the model 10 consecutive times...\n")

    """Remove infrequent words experiment"""
    # get a fresh model to serve as baseline to determine the frequency of words
    feature_vectors, correct_labels, _ = prepare_data(training_set)
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, DELTA)

    # conduct the experiment
    freq_list = (1, 5, 10, 15, 20)  # frequencies for which we remove words
    data_rem_infreq, x_ticks_infreq = infrequent_words_experiment(
        freq_list, nb_classifier, training_set, DELTA, test_vectors, test_correct_labels)

    """Remove most frequent words experiment"""

    # conduct experiment, using same baseline as previous series of experiments, because the model is intact
    freq_list = (0.05, 0.1, 0.15, 0.2, 0.25)  # frequencies we will remove
    data_rem_freq, x_ticks_freq = frequent_words_experiment(freq_list, nb_classifier, training_set, DELTA, test_vectors, test_correct_labels)

    # plot joint results
    initial_word_count = len(nb_classifier.model)
    plt.close()
    title = f"Frequency experiments\nInitial vocabulary {initial_word_count}"
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    multi_bar_plot(fig, ax[0], data_rem_infreq, x_ticks_infreq, "Removing Infrequent Words")
    multi_bar_plot(fig, ax[1], data_rem_freq, x_ticks_freq, "Removing Most Frequent Words")
    fig.suptitle(title)
    plt.subplots_adjust(left=0.18)
    plt.show(block=False)

    input("\nPress Enter to quit")

    print("\nBye!\n")


def load_data(csv_file_path):
    """

    :param csv_file_path:
    :return:
    """

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
    """

    :param remove_list:
    :param training_set:
    :param output_file:
    :param delta:
    :param word_filter:
    :return:
    """
    # train model
    feature_vectors, correct_labels, _ = prepare_data(training_set, word_filter)
    nb_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, delta)

    # print model stats and output model to file
    print_basic_model_stats(nb_classifier)
    output_model_to_file(nb_classifier, output_file)

    return nb_classifier


def testing_routine(nb_classifier, test_vectors, test_correct_labels, testing_set, output_file, t):
    """

    :param nb_classifier:
    :param test_vectors:
    :param test_correct_labels:
    :param testing_set:
    :param output_file:
    :param t:
    :return:
    """
    # test model, print stats and output results to file
    nb_scores, nb_labels, metrics = nb_classifier.test(test_vectors, test_correct_labels)
    print_metrics(metrics)
    output_test_results_to_file(nb_scores, nb_labels, test_correct_labels, testing_set, nb_classifier.classes, output_file)

    # plot results
    plt.close()
    x_ticks = ['accuracy']
    for class_name in nb_classifier.classes:
        x_ticks.append(f"{class_name}\nF1-score")

    data = [metrics.accuracy]
    for class_name in nb_classifier.classes:
        data.append(metrics.f1[class_name])

    fig, ax = plt.subplots(figsize=(7, 5))

    plt.bar(x_ticks, data, width=0.7)
    for i, v in enumerate(data):
        ax.text(i - 0.15, v + 0.01, "{:.4f}".format(v), fontweight='bold')

    ax.set_title(t)
    plt.show(block=False)


def prepare_data(data_set, filter_func=None, make_remove_list=False):
    """

    :param remove_list:
    :param data_set:
    :param filter_func:
    :return:
    """

    feature_vectors = []
    classifications = []
    removed_words = []

    # tokenize entries, count frequencies
    for row in data_set:

        class_name = row[3].strip()  # retrieve class name
        classifications.append(class_name)

        tokens = tokenize_row(row, removed_words, filter_func, make_remove_list)
        feature_vectors.append(tokens)

    return feature_vectors, classifications, removed_words


def tokenize_row(row, removed_words, filter_func, make_remove_list):
    """

    :param row:
    :param removed_words: list of removed words
    :param filter_func:
    :return:
    """
    punctuation = string.punctuation
    punctuation += "“”‘’«"
    title = row[2].replace('/', ' ')
    tokens = title.split()
    tokens = [token.strip(punctuation).lower() for token in tokens if token not in punctuation]
    tokens = [re.sub(r'[\'’]s$', '', token) for token in tokens]  # removing "'s" endings
    # remove words with digits and empty strings
    if not make_remove_list:
        tokens = [token for token in tokens if token and not re.match(r'.*\d.*', token)]
    else:
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
        tokens = filter_func(tokens)

    return tokens


def word_filter_factory(stop_words):
    """

    :param stop_words:
    :return:
    """

    def func(tokens):

        return [w for w in tokens if w not in stop_words]

    return func


def word_length_filter(tokens):
    """

    :param tokens:
    :param removed_words:
    :return:
    """

    return [w for w in tokens if not 2 < len(w) < 9]


def print_basic_model_stats(nb_classifier):
    """
    Prints out basic information about a model
    :param nb_classifier: the NaiveBayesClassifier object
    :return: None
    """
    print(f"size of vocabulary: {len(nb_classifier.model)}")
    print(f'classification counts: {nb_classifier.class_frequencies}')
    print(f'words per class: {nb_classifier.word_frequencies}')
    print(f'Priors: {nb_classifier.class_prior_likelihoods}')
    print()


def print_metrics(metrics):
    """
    Prints out data of a Metrics object
    :param metrics: Metrics object
    :return: None
    """
    print("TEST METRICS\n")
    print("Accuracy: {0:.4f}".format(metrics.accuracy))
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.precision.items()]
    print(f"Precision: {''.join(line)}")
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.recall.items()]
    print(f"Recall: {''.join(line)}")
    line = [" '{}': {:.4f},".format(class_name, value) for class_name, value in metrics.f1.items()]
    print(f"F1 Scores: {''.join(line)}")
    print()


def output_model_to_file(classifier, file_name):
    """
    Formats and ouputs the model to file.
    :param classifier: the NaiveBayesClassifier object
    :param file_name: the output file name
    :return: None
    """
    # prepare model output file
    model_lines = []
    line_counter = 1

    for feature, feature_properties in sorted(classifier.model.items()):

        line = [f'{line_counter}  {feature}']  # using a list to avoid multiple concatenations
        for class_name in classifier.classes:
            line.append(
                f'  [{class_name}'
                f'  {feature_properties.frequencies[class_name]}'
                f'  {"{0:.9f}".format(feature_properties.likelihoods[class_name])}]')
        line.append('\n')

        model_lines.append(''.join(line))

        line_counter += 1

    model_file_name = f'{file_name}.txt'

    with open(model_file_name, "w", encoding='utf-8') as model_file:
        model_file.writelines(model_lines)


def output_test_results_to_file(nb_scores, nb_labels, test_correct_labels, testing_set, classes, file_name):
    """
    Outputs results of testing a model to file
    :param nb_scores: list returned by the NaiveBayesClassifier, which contains
    probabilities of each class for each instance
    :param nb_labels: list of labels returned by the NaiveBayesClassifier
    :param test_correct_labels: list of correct labels of the testing sample
    :param testing_set: list of vectors of features which was originally fed to the classifier
    :param classes: the list of the classes of the classifier
    :param file_name: output file name
    :return: None
    """
    lines = []
    line_counter = 1

    # outputting results to file
    for (nb_score_vector, nb_label, correct_label, row) in zip(nb_scores, nb_labels, test_correct_labels, testing_set):

        right_wrong = "right" if nb_label == correct_label else "wrong"

        line = [f'{line_counter}  {row[2]}  {nb_label}']  # using a list to avoid multiple concatenations
        for i, class_name in enumerate(classes):
            line.append(f'  [{class_name} {"{0:.6f}".format(nb_score_vector[i])}]')
        line.append(f'  {correct_label}  {right_wrong}\n')
        lines.append(''.join(line))
        line_counter += 1

    file_name = f'{file_name}.txt'

    with open(file_name, "w", encoding='utf-8') as file:
        file.writelines(lines)


def infrequent_words_experiment(freq_list, nb_classifier, training_set, delta, test_vectors, test_correct_labels):
    """
    Encapsulates the experiment of removing sequentially the words with the least frequency,
    iterating through to a list of frequency values, and return the combined results of the experiment.
    A baseline classifier is used to determine lists of words to remove for each iterations
    :param delta:
    :param training_set:
    :param freq_list: list of frequencies for which we remove words in the model.
    A new model is then trained for each iteration.
    :param nb_classifier: the baseline NaiveBayesClassifier used to determine removal lists base on
    frequency of words
    :param test_vectors: the vectors containing the test inputs
    :param test_correct_labels: the correct labels for the test inputs
    :return: accuracy and F1-score for each class, for each frequency experiment, the list of ticks to be used
    to label the x-axis on a plot (corresponding to the number of words remaining in the model after each experiment).
    """
    # to stock metrics for each round of removal
    # for the purposes of plotting
    data_remove_infreq = {'accuracy': []}
    for class_name in nb_classifier.classes:
        data_remove_infreq[class_name] = []

    x_ticks_infreq = []  # x-axis of plotting: number of words remaining in vocabulary

    # getting metrics after each round of removal
    for frequency in freq_list:
        stop_words = set()
        for word, properties in nb_classifier.model.items():
            if sum(properties.frequencies.values()) <= frequency:
                stop_words.add(word)

        # get a word filter for excluded words and train a new model
        word_filter = word_filter_factory(stop_words)

        feature_vectors, correct_labels, _ = prepare_data(training_set, word_filter)
        nb_temp_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, delta)
        _, _, metrics = nb_temp_classifier.test(test_vectors, test_correct_labels)

        data_remove_infreq['accuracy'].append(metrics.accuracy)
        for class_name in nb_classifier.classes:
            data_remove_infreq[class_name].append(metrics.f1[class_name] if class_name in metrics.f1 else 0)
        
        x_ticks_infreq.append(len(nb_temp_classifier.model))

    return data_remove_infreq, x_ticks_infreq


def frequent_words_experiment(freq_list, nb_classifier, training_set, delta, test_vectors, test_correct_labels):
    """
    Encapsulates the experiment of removing sequentially the most frequent words,
    according to a list of frequency percentages, and return the combined results of the experiment.
    :param delta:
    :param training_set:
    :param freq_list: list of frequencies for which we remove words in the model
    :param nb_classifier: the NaiveBayesClassifier
    :param test_vectors: the vectors containing the test inputs
    :param test_correct_labels: the correct labels for the test inputs
    :return: accuracy and F1-score for each class, for each frequency experiment, the list of ticks to be used
    to label the x-axis on a plot (corresponding to the number of words remaining in the model after each experiment).
    """

    # to stock metrics for each round of removal
    data_remove_freq = {'accuracy': []}
    for class_name in nb_classifier.classes:
        data_remove_freq[class_name] = []

    x_ticks_freq = []  # x-axis for plotting, number of words remaining in vocabulary after each experiment

    # order keys in terms of frequency
    sorted_freq_keys = sorted(nb_classifier.model.keys(),
                              key=lambda k: sum(nb_classifier.model[k].frequencies.values()),
                              reverse=True)

    for frequency in freq_list:
        # get most frequent keys and add to remove list
        stop_words = set()
        for i in range(math.floor(frequency * len(sorted_freq_keys))):
            stop_words.add(sorted_freq_keys[i])

        # train new model without the removed words
        # get a word filter for excluded words and train a new model
        word_filter = word_filter_factory(stop_words)
        feature_vectors, correct_labels, _ = prepare_data(training_set, word_filter)
        nb_temp_classifier = NaiveBayesClassifier(feature_vectors, correct_labels, delta)
        _, _, metrics = nb_temp_classifier.test(test_vectors, test_correct_labels)

        # populate data fields following classes from the baseline model (for uniformity)
        # Then, if one model doesn't have all classes, we add 0 by default
        data_remove_freq['accuracy'].append(metrics.accuracy)
        for class_name in nb_classifier.classes:
            data_remove_freq[class_name].append(metrics.f1[class_name] if class_name in metrics.f1 else 0)

        x_ticks_freq.append(len(nb_temp_classifier.model))

    return data_remove_freq, x_ticks_freq


def multi_bar_plot(fig, ax, data, x_ticks, title, legend=True):
    """
    Builds a multi-bar graph, multiple data points

    :param fig:
    :param ax:
    :param data: a dictionary mapping the names of each bar, to a list of the values
    for all the data points
    :param x_ticks: the labels for each multi-bar data point
    :param title:
    :param legend: indicates whether to add or legend for the different bars for each data point
    :return: None
    """

    total_width = 0.8
    single_width = 0.9

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
        fig.legend(bars, [f'F1 score: {key}' for key in data.keys()], loc='center left')

    x_ticks.insert(0, None)
    ax.set_xticklabels(x_ticks)
    ax.set_title(title)
    ax.set_xlabel('words remaining')


if __name__ == "__main__":
    main()

