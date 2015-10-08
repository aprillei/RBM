#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "rbm_util.h"
#include "rbm.h"
#include "paras.h"
#include "util.h"

/**
* Binary Restricted Boltzmann Machine using k-step Contrastive Divergence.
* Refs:
*   Fischer, Asja, and Christian Igel. "Training restricted Boltzmann machines: an introduction." Pattern Recognition 47.1 (2014): 25-39.
*   Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." Momentum 9.1 (2010): 926.
*   Chang, Chih-Chung, and Chih-Jen Lin. "LIBSVM: A library for support vector machines." ACM Transactions on Intelligent Systems and Technology (TIST) 2.3 (2011): 27.
*   http://deeplearning.net/tutorial/rbm.html
*   20 News Data set http://qwone.com/~jason/20Newsgroups/
*/

int main(int argc, char * argv[]){
    srand(time(NULL));
    // argv[1]: train file (eg. train for train.fea and train.label)
    // argv[2]: test file
    // argv[3]: NUM_TRAIN (2245)
    // argv[4]: NUM_TEST (1494)
    // argv[5]: NUM_VISIBLE_UNITS (1000)
    // argv[6]: NUM_HIDDEN_UNITS (100)
    // argv[7]: LEARNING_RATE (0.1)
    // argv[8]: NUM_ITERATIONS
    // argv[9]: K
    // argv[10]: BATCH_SIZE
    // argv[11]: paras.para

    std::string train = argv[1];   //"train_trunc"
    std::string test = argv[2];  //"test_trunc"
    int NUM_TRAIN = std::stoi(argv[3]);
    int NUM_TEST = std::stoi(argv[4]);
    int NUM_VISIBLE_UNITS = std::stoi(argv[5]);
    int NUM_HIDDEN_UNITS = std::stoi(argv[6]);
    float LEARNING_RATE = std::stof(argv[7]);
    int NUM_ITERATIONS = std::stoi(argv[8]);
    int K = std::stoi(argv[9]);
    int BATCH_SIZE = std::stoi(argv[10]);
    std::string paras_file = argv[11];

    // generate libsvm features for raw features
    //std::cout << "Formatting raw features into libsvm format" << std::endl;
    transform_to_libsvm_format(train, NUM_TRAIN, NUM_VISIBLE_UNITS, "learned_features/train.svm");
    transform_to_libsvm_format(test, NUM_TEST, NUM_VISIBLE_UNITS, "learned_features/test.svm");

    // generate hidden features from train/test sets
    //std::cout << "Train RBM" << std::endl;
    std::string model_weight_file_train = "parameters/weights_train";
    std::string model_visible_bias_file_train = "parameters/visible_bias_train";
    std::string model_hidden_bias_file_train = "parameters/hidden_bias_train";
    std::string learned_features_train = "learned_features/learned_features_train";
    std::string learned_features_test = "learned_features/learned_features_test";

    rbm_paras para;
    load_rbm_paras(para, paras_file.c_str());
    rbm myrbm1(para, NUM_TRAIN);
    myrbm1.run(model_weight_file_train, model_visible_bias_file_train, model_hidden_bias_file_train, learned_features_train, train);

    myrbm1.generate_features(train, model_weight_file_train.c_str(),
            model_visible_bias_file_train.c_str(), model_hidden_bias_file_train.c_str(),
            learned_features_train.c_str(), NUM_TRAIN);

    rbm myrbm2(para, NUM_TEST);
    myrbm2.generate_features(test, model_weight_file_train.c_str(),
            model_visible_bias_file_train.c_str(), model_hidden_bias_file_train.c_str(),
            learned_features_test.c_str(), NUM_TEST);

    // generate libsvm features for deeplearning.org implementation
    //std::cout << "Training deeplearning.net RBM" << std::endl;
    std::string model_weight_file = "parameters/weights_dl_implementation";
    std::string model_visible_bias_file = "parameters/visible_bias_dl_implementation";
    std::string model_hidden_bias_file = "parameters/hidden_bias_dl_implementation";
    std::string learned_features_file_train = "learned_features/learned_features_dl_implementation_train";
    std::string learned_features_file_test = "learned_features/learned_features_dl_implementation_test";

    generate_features_DL_Implementation(train, model_weight_file.c_str(),
                model_visible_bias_file.c_str(), model_hidden_bias_file.c_str(),
                learned_features_file_train.c_str(), NUM_TRAIN, NUM_VISIBLE_UNITS, NUM_HIDDEN_UNITS);

    generate_features_DL_Implementation(test, model_weight_file.c_str(),
                model_visible_bias_file.c_str(), model_hidden_bias_file.c_str(),
                learned_features_file_test.c_str(), NUM_TEST, NUM_VISIBLE_UNITS, NUM_HIDDEN_UNITS);

    return 0;
}

