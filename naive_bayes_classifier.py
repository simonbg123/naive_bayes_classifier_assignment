import math


class NaiveBayesClassifier:
    """

    """

    def __init__(self, feature_vectors, correct_labels, delta):
        """

        :param feature_vectors:
        :param correct_labels:
        :param delta:
        """
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
        """

        :param vectors:
        :return:
        """
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
        """

        :param test_vectors:
        :param test_correct_labels:
        :return:
        """
        nb_scores, nb_labels = self.classify(test_vectors)
        metrics = self.get_metrics(nb_labels, test_correct_labels)

        return nb_scores, nb_labels, metrics

    def get_metrics(self, nb_labels, correct_labels):
        """

        :param nb_labels:
        :param correct_labels:
        :return:
        """

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
        """

        :return:
        """
        return [word for word in sorted(self.model.keys())]


class FeatureProperties:
    """
    This class is used to hold overall frequency and
    conditional probabilities with respect to each class
    for each feature key in the model
    """
    def __init__(self, classes):

        # frequencies
        self.frequencies = {class_name: 0 for class_name in classes}

        # conditional probabilities
        self.likelihoods = {class_name: 0 for class_name in classes}


class Metrics:
    """
    Used to return metrics after testing a model.
    It holds the general accuracy of the model,
    as well as precision, recall and F1-measure for
    each class.
    """

    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

