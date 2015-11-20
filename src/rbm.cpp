#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <math.h>
#include "rbm.h"
#include "rbm_util.h"
#include "util.h"
#include <random>

rbm::rbm(rbm_paras para, int n_data_points){
    num_visible_units = para.num_visible_units;
    num_hidden_units = para.num_hidden_units;
    K = para.K;
    num_epochs = para.num_epochs;
    learning_rate = para.learning_rate;
    size_minibatch = para.size_minibatch;
    number_of_data_points = n_data_points;
}

void rbm::load_data(std::string data_file){
    // Load in binary format:
    // if feature == 0: 0 , 1 otherwise
    std::string feature_file(data_file);
	std::string label_file(data_file);
	feature_file += ".fea";
	label_file += ".labels";

    input_features = new float*[number_of_data_points];
    for (int i=0; i<number_of_data_points; i++){
        input_features[i] = new float[num_visible_units];
    }

    std::ifstream infile(feature_file.c_str(),std::ios::in);
    for (int i=0; i<number_of_data_points; i++){
        for (int j=0; j<num_visible_units; j++){
            infile >> input_features[i][j];
            // Binary RBM
            if (input_features[i][j] > 0){
                input_features[i][j] = 1;
            }
        }
    }
    infile.close();

    int * output_labels = new int[number_of_data_points];
    FILE * tmpfp = fopen(label_file.c_str(),"r");
	for (int i = 0; i < number_of_data_points; i++){
		fscanf(tmpfp, "%d", &output_labels[i]);
	}
	fclose(tmpfp);
}

// forward activation
void rbm::forward_activation(float ** weights, float * hidden_bias, float * visible, float * hidden){
    multiply_matrix_vector(weights, visible, hidden, num_hidden_units, num_visible_units);
    add_vector(hidden, hidden_bias, num_hidden_units);
    activate_logistic(hidden, num_hidden_units);
}

// randomly initialize weights and biases
//void rbm::init_paras(float ** weights, float * visible_bias, float * hidden_bias, const char * init_weights_file, const char * init_hidden_bias_file, const char * init_visible_bias_file){
void rbm::init_paras(float ** weights, float * visible_bias, float * hidden_bias){
    srand(time(NULL));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.01);

    for (int i=0; i<num_hidden_units; i++){
        for (int j=0; j<num_visible_units; j++){
            weights[i][j] = distribution(generator);
        }
    }

    // Initialize the hidden bias
    for (int i=0; i<num_hidden_units; i++){
        hidden_bias[i] = randfloat();
    }

    // Initialize the visible bias
    for (int i=0; i<num_visible_units; i++){
        visible_bias[i] = randfloat();
    }
}

// train model
void rbm::train(float ** weights, float * visible_bias, float * hidden_bias){
    std::cout << "Training model..." << std::endl;

    // weights, visible_bias and hidden_bias were randomly initialized
    // Initialize delta_weights, delta_hidden_bias, and delta_visible_bias to 0s
    float ** delta_weights = new float*[num_hidden_units];
    for (int i=0; i<num_hidden_units; i++){
        delta_weights[i] = new float[num_visible_units];
    }
    float * delta_hidden_bias = new float[num_hidden_units];
    float * delta_visible_bias = new float[num_visible_units];

    float ** features = new float*[number_of_data_points];
    for (int i=0; i<number_of_data_points; i++){
        features[i] = new float[num_visible_units];
    }

    for (int i=0; i<number_of_data_points; i++){
        for (int j=0; j<num_visible_units; j++){
            features[i][j] = input_features[i][j];
        }
    }

    // random indexes of the data points that will be chosen at each iteration of sga
    int * idxes_batch = new int[size_minibatch];

	int inner_iter = number_of_data_points/num_epochs;
    // Perform K-step cd num_epochs time
    for (int iter=0; iter<num_epochs; iter++){
		for (int i=0; i<inner_iter; i++){
       		// sample minibatch and perform cd on this mini batch
        	// sample minibatch
        	rand_init_vec_int(idxes_batch, size_minibatch, number_of_data_points);

        	// set deltas to zeros at every iteration in cd
        	cd(features, weights, hidden_bias, visible_bias, delta_weights, delta_hidden_bias,
            	    delta_visible_bias, K, idxes_batch, size_minibatch);

        	// update parameters
        	update_weights(weights, delta_weights, learning_rate, num_hidden_units, num_visible_units, size_minibatch);
        	update_visible_bias(visible_bias, delta_visible_bias, learning_rate, num_visible_units, size_minibatch);
        	update_hidden_bias(hidden_bias, delta_hidden_bias, learning_rate, num_hidden_units, size_minibatch);
		}
    }

    // release data
    delete[] idxes_batch;
    delete[] delta_hidden_bias;
    delete[] delta_visible_bias;

    for (int i=0; i<num_hidden_units; i++){
        delete[] delta_weights[i];
    }
    delete[] delta_weights;

    for (int i=0; i<number_of_data_points; i++){
        delete[] features[i];
    }
    delete[] features;
//    std::cout << "Training model: DONE" << std::endl;
}

// run the whole learning process of RBM
void rbm::run(std::string model_weight_file, std::string model_visible_bias_file,
            std::string model_hidden_bias_file, std::string learned_features_file, std::string file){

    load_data(file.c_str());

    // randomly initialize weights and biases
    float ** weights = new float*[num_hidden_units];
    float * hidden_bias = new float[num_hidden_units];
    float * visible_bias = new float[num_visible_units];

    for (int i=0; i<num_hidden_units; i++){
        weights[i] = new float[num_visible_units];
    }

    init_paras(weights, visible_bias, hidden_bias);
//	std::string in_weights = "init_weights";
//	std::string in_hidden = "init_hidden";
//	std::string in_visible = "init_visible";
//    init_paras(weights, visible_bias, hidden_bias, in_weights.c_str(), in_hidden.c_str(), in_visible.c_str());

    // train rbm
    train(weights, visible_bias, hidden_bias);

    // save model
    save_model(weights, visible_bias, hidden_bias, model_weight_file.c_str(),
                model_visible_bias_file.c_str(), model_hidden_bias_file.c_str());

    for (int i=0; i<num_hidden_units; i++){
        delete[] weights[i];
    }
    delete[] weights;

    delete[] visible_bias;
    delete[] hidden_bias;
}

// Gradient ascent update for the weights
void rbm::update_weights(float ** weights, float ** delta_weights, float learning_rate, int n, int m, float size_mBatch){
    multiply_constant_matrix(delta_weights,learning_rate, n, m);
    add_matrix(weights, delta_weights, n, m);
}

// Gradient ascent update for the visible bias
void rbm::update_visible_bias(float * visible_bias, float * delta_visible_bias, float learning_rate, int m, float size_mBatch){
    multiply_constant_vector(delta_visible_bias,learning_rate,m);
    add_vector(visible_bias, delta_visible_bias,m);
}

// Gradient ascent update for the hidden bias
void rbm::update_hidden_bias(float * hidden_bias, float * delta_hidden_bias, float learning_rate, int n, float size_mBatch){
    multiply_constant_vector(delta_hidden_bias, learning_rate, n);
    add_vector(hidden_bias,delta_hidden_bias,n);
}

// TODO add a isPersistent attribute to distinguish
// between cd and pcd (initialize from the old state of the chain)
// TODO compute pseudo likelihood
// S: Sample -> Indexes of the elements in the sample considered
// size(S) = size_mBatch
//void rbm::cd(float ** weights, float * hidden_bias, float * visible_bias, float ** delta_weights, float * delta_hidden_bias,
 //       float * delta_visible_bias, int K, int * S, int size_mBatch){
void rbm::cd(float ** features, float ** weights, float * hidden_bias, float * visible_bias, float ** delta_weights, float * delta_hidden_bias,
        float * delta_visible_bias, int K, int * S, int size_mBatch){
    // initialize delta_weights and delta_biases to 0s
    for (int i=0; i<num_hidden_units; i++){
        memset(delta_weights[i], 0, sizeof(float)*num_visible_units);
    }
    memset(delta_hidden_bias, 0, sizeof(float)*num_hidden_units);
    memset(delta_visible_bias, 0, sizeof(float)*num_visible_units);

    for (int i=0; i<size_mBatch; i++){
        cd_single_data(features[S[i]], weights, hidden_bias, visible_bias, delta_weights,
                    delta_hidden_bias, delta_visible_bias, K);
    }
}

void rbm::cd_single_data(float * features_idx, float ** weights, float * hidden_bias, float * visible_bias, float ** delta_weights,
                    float * delta_hidden_bias, float * delta_visible_bias, int K){

    float * v = new float[num_visible_units];
    float * h = new float[num_hidden_units];

    float * p_h = new float[num_hidden_units];
    float * p_v = new float[num_visible_units];
    float * p_0 = new float[num_hidden_units];

    float * wvc = new float[num_hidden_units];
    float * whb = new float[num_visible_units];

    copy_vec(v, features_idx, num_visible_units);

    // compute wv0
    multiply_matrix_vector(weights, v, wvc, num_hidden_units, num_visible_units);

    // compute wvc0
    add_vector(wvc, hidden_bias, num_hidden_units);

    // compute p_0
    compute_probability_hidden_given_visible(wvc, p_0, num_hidden_units);

    // do K-step gibbs sampling
    for (int i=0; i<K; i++){
        // compute wv
        multiply_matrix_vector(weights, v, wvc, num_hidden_units, num_visible_units);

        // compute wvc
        add_vector(wvc, hidden_bias, num_hidden_units);

        compute_probability_hidden_given_visible(wvc, p_h, num_hidden_units);
        sample_vector(h, p_h, num_hidden_units);

        // compute wh
        multiply_matrix_vector_column_wise(weights, h, whb, num_hidden_units, num_visible_units);

        // compute whb
        add_vector(whb, visible_bias, num_visible_units);

        compute_probability_visible_given_hidden(whb, p_v, num_visible_units);
        sample_vector(v, p_v, num_visible_units);
    }

    update_weights_gradient(delta_weights, features_idx, v, p_0, p_h, num_hidden_units, num_visible_units);
    update_visible_bias_gradient(delta_visible_bias, features_idx, v, num_visible_units);
    update_hidden_bias_gradient(delta_hidden_bias, p_0, p_h, num_hidden_units);

    delete[] v;
    delete[] h;

    delete[] p_h;
    delete[] p_v;
    delete[] p_0;

    delete[] wvc;
    delete[] whb;
}

// save model
void rbm::save_model(float ** weights, float * visible_bias, float * hidden_bias, const char * mdl_weight_file,
                const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file){
    // Save weights, and biases
    // save visible bias
    std::ofstream visible_bias_file(mdl_visible_bias_file);
    if (visible_bias_file.is_open()){
        for (int i=0; i<num_visible_units; i++){
            visible_bias_file << visible_bias[i] << std::endl;
        }
        visible_bias_file.close();
    }

    // save hidden bias
    std::ofstream hidden_bias_file(mdl_hidden_bias_file);
    if (hidden_bias_file.is_open()){
        for (int i=0; i<num_hidden_units; i++){
            hidden_bias_file << hidden_bias[i] << std::endl;
        }
        hidden_bias_file.close();
    }

    // save weights
    std::ofstream weights_file(mdl_weight_file);
    if (weights_file.is_open()){
        for (int i=0; i<num_hidden_units; i++){
            for (int j=0; j<num_visible_units; j++){
                weights_file << weights[i][j] << " ";
            }
            weights_file << std::endl;
        }
    }
}

// Given learned parameters and new input vectors, compute the hidden nodes
// and save them in a libsvm input format (learned_features)
void rbm::generate_features(std::string data_file, const char * mdl_weight_file,
                const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file,
                const char * learned_features_file, int number_of_examples){
    std::string feature_file(data_file);
	std::string label_file(data_file);
	feature_file += ".fea";
	label_file += ".labels";

    // load Weights into weights matrix
    float ** weights = new float*[num_hidden_units];
    for (int i=0; i<num_hidden_units; i++){
        weights[i] = new float[num_visible_units];
    }
    std::ifstream weight_file(mdl_weight_file,std::ios::in);
    for (int i=0; i<num_hidden_units; i++){
        for (int j=0; j<num_visible_units; j++){
            weight_file >> weights[i][j];
        }
    }
    weight_file.close();

    // load hidden bias into hidden bias vector
    float * hidden_bias = new float[num_hidden_units];
    std::ifstream hidden_bias_file(mdl_hidden_bias_file,std::ios::in);
    for (int i=0; i<num_hidden_units; i++){
        hidden_bias_file >> hidden_bias[i];
    }
    hidden_bias_file.close();

    float ** input_f = new float*[number_of_examples];
    for (int i=0; i<number_of_examples; i++){
        input_f[i] = new float[num_visible_units];
    }
    std::ifstream feat_file(feature_file.c_str(),std::ios::in);
    for (int i=0; i<number_of_examples; i++){
        for (int j=0; j<num_visible_units; j++){
            feat_file >> input_f[i][j];
            // Binary RBM
            if (input_f[i][j] > 0){
                input_f[i][j] = 1;
            }
        }
    }
    feat_file.close();

    int * output_labels = new int[number_of_examples];
    FILE * tmpfp = fopen(label_file.c_str(),"r");
	for (int i = 0; i < number_of_examples; i++){
		fscanf(tmpfp, "%d", &output_labels[i]);
	}
	fclose(tmpfp);

    float ** hidden = new float*[number_of_examples];
    for (int i=0; i<number_of_examples; i++){
        hidden[i] = new float[num_hidden_units];
    }

    // for each example, compute the corresponding hidden features
    for (int i=0; i<number_of_examples; i++){
        compute_hidden(weights, hidden_bias, input_f[i], hidden[i], num_hidden_units, num_visible_units);
    }

    // write the learned features to learned_features
    // libsvm uses a sparse format
    std::ofstream out_file(learned_features_file);
    if (out_file.is_open()){
        for (int i=0; i<number_of_examples; i++){
            int non_zero_features = 0;
            for (int j=0; j<num_hidden_units; j++){
//                if (hidden[i][j] > 0.0001 || hidden[i][j] < -0.0001){ // TODO temp/ Should be != 0
                if (hidden[i][j] != 0){
                    non_zero_features++;
                    if (non_zero_features == 1){
                        out_file << output_labels[i];
                    }
                    out_file << " " << (j+1) << ":" << hidden[i][j];    // indexing starts at 1 in libsvm
                }
            }
            if (non_zero_features > 0){
                out_file << std::endl;
            }
        }
    }
    out_file.close();

    //release data
    delete[] input_f;
}



