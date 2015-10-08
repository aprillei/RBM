# include "rbm_util.h"
# include "util.h"
#include <fstream>
#include <iostream>
// TODO Add momentum and weight decay

// update visible bias in k-step contrastive divergence
// delta_b = delta_b + v_0 - v_k
// dim(delta_b) = m x 1
// b denotes the hidden bias
void update_visible_bias_gradient(float * delta_b, float * v_0, float * v_k, int m){
    for (int i=0; i<m; i++){
        delta_b[i] = delta_b[i] + v_0[i] - v_k[i];
    }
}

// update hidden bias in k-step contrastive divergence
// delta_c = delta_c + p_0 - p_k
// where p_0[i] = P(H_{i} = 1 | v_0)
// where p_k[i] = P(H_{i} = 1 | v_k)
// dim(delta_c) = n x 1
// c denotes the visible bias
void update_hidden_bias_gradient(float * delta_c, float * p_0, float * p_k, int n){
    for (int i=0; i<n; i++){
        delta_c[i] = delta_c[i] + p_0[i] - p_k[i];
    }
}

// update the weigths gradient in k-step contrastive divergence
// dim(delta_w) = n x m
// dim(v_0) = dim(v_k) = m x 1
// dim(p_0) = dim(p_k) = n x 1
void update_weights_gradient(float ** delta_w, float * v_0, float * v_k, float * p_0, float * p_k, int n, int m){
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            delta_w[i][j] = delta_w[i][j] + p_0[i]*v_0[j] - p_k[i]*v_k[j];
        }
    }
}

// compute P(H_{i} = 1 | v)
// wvc is the sum of wv and c - the visible bias
void compute_probability_hidden_given_visible(float * wvc, float * p, int n){
    for (int i=0; i<n; i++){
        p[i] = logistic(wvc[i]);
    }
}

// compute P(V_{j} = 1 | h)
// dim(whb) = m x 1
void compute_probability_visible_given_hidden(float * whb, float * p, int m){
    for (int j=0; j<m; j++){
        p[j] = logistic(whb[j]);
    }
}

// sample h_{i}^{t} ~ p(h_{i} | v^{t})
// or v_{j}^{t} ~ p(v_{j} | h^{t})
// i=1..n
// j=1..m
void sample_vector(float * vec, float * p, int n){
    for (int i=0; i<n; i++){
        vec[i] = p[i];
    //    vec[i] = ((p[i] > randfloat()) ? 1 : 0);
    }
}

// compute hidden nodes given model parameters and for a new visible input
void compute_hidden(float ** weights, float * hidden_bias, float * visible, float * hidden, int num_hidden_units, int num_visible_units){
    multiply_matrix_vector(weights, visible, hidden, num_hidden_units, num_visible_units);
    add_vector(hidden, hidden_bias, num_hidden_units);
    activate_logistic(hidden, num_hidden_units);

}


// Given learned parameters and new input vectors, compute the hidden nodes
// and save them in a libsvm input format (learned_features)
// This method is used for the output of the persistent model (deeplearning.net)
// for which the weights matrix has dimensions num_visible x num_hidden
void generate_features_DL_Implementation(std::string data_file, const char * mdl_weight_file,
                const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file,
                const char * learned_features_file, int number_of_examples, int num_visible_units, int num_hidden_units){
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
    for (int i=0; i<num_visible_units; i++){
        for (int j=0; j<num_hidden_units; j++){
            weight_file >> weights[j][i];
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
                if (hidden[i][j] != 0){
//                if (hidden[i][j] > 0.0001 || hidden[i][j] < -0.0001){ // TODO temp/ Should be != 0
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
}

