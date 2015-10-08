__author__ = 'Hakim'

'''
Input:
train.fea, train.label, test.fea, test.label text files

Label:
One row for each example
label1
label2
etc

Features:
One row for each example
fea_1_1 fea_1_2 etc (features are space delimited)


Output:
Pickled file containing
train_set, test_set format: tuple(np_input, np_target)
    np_input is a numpy.ndarray of 2 dimensions (a matrix)
    row's correspond to an example.
    np_target is a numpy.ndarray of 1 dimension)
    that has same length as the number of rows in np_input
    np_target gives the target to the example with the same index in the np_input
'''

import numpy as np
import cPickle

def pkl_set(features_file, labels_file):
    # save all the features in a list
    all_features = []

    # save all the labels in a list
    all_labels = []

    # save the features for each row in a list
    current_features = []

    # read each row of the features file and save the features in a list
    with open(features_file) as f_file:
        for line in f_file:
            line = line.split(' ')
            line.remove('\n')
            current_features = map(int, line)
            all_features.append(list(current_features))

    # read each row of the labels file and save the corresponding label
    with open(labels_file) as l_file:
        for line in l_file:
            line = line.strip('\n')[0]
            all_labels.append(map(int, line)[0])

    # transform all_features into a numpy ndarray
    features_array = np.array(all_features)

    # transform all_labels into a numpy ndarray
    labels_array = np.array(all_labels)

    # return tuple
    return features_array, labels_array

def pkl_all(training_features_file, training_labels_file,
            test_features_file, test_labels_file):

    to_dump = []
    train_tuple = pkl_set(training_features_file, training_labels_file)
    test_tuple = pkl_set(test_features_file, test_labels_file)

    to_dump.append(train_tuple)
    to_dump.append(test_tuple)

    cPickle.dump(to_dump, open('dataset/20_news_data.pkl','wb'))

if __name__ == '__main__':
    from sys import argv
    pkl_all(argv[1], argv[2], argv[3], argv[4])