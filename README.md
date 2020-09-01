# Naive Bayes Classifier, AI Assignment
Author: Simon Brillant-Giroux


## Required Libraries (other than modules from the standard library):

matplotlib

## Custom modules which are included in this subsmission:
	
naive_bayes_classifier

## Included
	
- run_experiment.py
- naive_bayes_classifier.py
- csv sample input file: hns_2018_2019.csv
- sample_output/    
	>*This includes all output files from an example run on the assignment data.*

## Usage:

	nb_classifier [csv_file_path]

To use this program the run_experiment.py script with the .csv input file as argument.

Ex: python nb_classifier.py hns_2018_2019.csv

The program will execute each task in the assignment. When the program has finished a task, it will show a plot and prompt the user to press Enter to continue on to the next task.

## Description

Basic model stats are printed to the terminal. If the training data did not contain any testing data (2019 entries), a message will be printed and tests will not be conducted. If tests are conducted, test statistics are printed to the terminal. All required output text files (model files and result files) are created in the output directory. 

Model files are text representations of the trained model. They contain a list of features and their frequencies and likelihoods given each class that was present is the training phase. The order of the classes is shown at the first line of the file. Result files contain each entry in the test sample, along with their NB score for each class, the resulting classification, and weather the classification was correct. For the initial baseline model, a vocabulary list and removed-word list are also printed.

A plot result plot is also shown on the monitor, showing global accuracy of the model, as well as F1 scores for each class.

For the frequency experiments, results of the multiple stages of the experiments are combined in bar charts. These experiments consist in experimenting with word removal based on frequency, in increments.

This assignment was completed with my own Naive Bayes Classifier module. Words are removed for the baseline model consisting of words containing numeric symbols. Tokens are trimmed of punctuation symbols and the special case of the "'s" endings. Those words are added in the remove_words.txt file.
