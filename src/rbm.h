#ifndef RBM_H_INCLUDED
#define RBM_H_INCLUDED

#include "paras.h"

// class of Restricted Boltzmann Machines
class rbm{
    private:
        // meta parameters
        int num_visible_units;
        int num_hidden_units;
        int stepsize;   // stepsize of SGD
        int size_minibatch;    // size of a minibatch
        int K;  // number of cd steps
        int num_epochs;     // number of sgd iterations
        float learning_rate;    // sgd learning rate

        // data
        float ** input_features;    // input features
        int * output_labels;    // output labels
        int number_of_data_points; //number of data points

    public:
        // constructor
        rbm(rbm_paras para, int n_data_points);

        //
        float ** get_input_features();
        // load input data
        void load_data(std::string data_file);

        // forward activation
        void forward_activation(float ** weights, float * hidden_bias, float * visible, float * hidden);

        // randomly initialize weights and biases
        void init_paras(float ** weights, float * visible_bias, float * hidden_bias);
//		void init_paras(float ** weights, float * visible_bias, float * hidden_bias, const char * init_weights_file, const char * init_hidden_bias_file, const char * init_visible_bias_file);

        // train model
        void train(float ** weights, float * visible_bias, float * hidden_bias);

        // run the whole learning process of RBM
        void run(std::string model_weight_file, std::string model_visible_bias_file,
                std::string model_hidden_bias_file, std::string learned_features_file, std::string file);

        // Gradient ascent update for the weights
        // dim(weights) = m x n
        void update_weights(float ** weights, float ** delta_weights, float learning_rate, int n, int m, float size_mBatch);

        // Gradient ascent update for the visible bias
        // dim(visible_bias) = m x 1
        void update_visible_bias(float * visible_bias, float * delta_visible_bias, float learning_rate, int m, float size_mBatch);

        // Gradient ascent update for the hidden bias
        //dim(hidden_bias) = n x 1
        void update_hidden_bias(float * hidden_bias, float * delta_hidden_bias, float learning_rate, int n, float size_mBatch);

        // k-step cd on size_mBatch input samples given by S indexes
        void cd(float ** features, float ** weigths, float * hidden_bias, float * visible_bias,
                float ** delta_weigths, float * delta_hidden_bias,
                float * delta_visible_bias, int k, int * S, int size_mBatch);

        // k-step cd on a single data point
        void cd_single_data(float * features_idx, float ** weigths, float * hidden_bias, float * visible_bias,
                            float ** delta_weigths, float * delta_hidden_bias, float * delta_visible_bias, int k);

        // save model
        void save_model(float ** weights, float * visible_bias, float * hidden_bias, const char * mdl_weight_file,
                        const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file);

        // Given learned parameters and a new input vector, compute the hidden nodes
        // and save them in a libsvm input format http://www.csie.ntu.edu.tw/~r94100/libsvm-2.8/README
        void generate_features(std::string data_file, const char * mdl_weight_file,
                const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file,
                const char * learned_features, int number_of_examples);
};


#endif // RBM_H_INCLUDED
