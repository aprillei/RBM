#ifndef RBM_UTIL_H_INCLUDED
#define RBM_UTIL_H_INCLUDED

# include <cmath>
# include <string>

// update the visible bias in k-step contrastive divergence
void update_visible_bias_gradient(float * delta_b, float * v_0, float * v_k, int m);

// update the hidden bias in k-step contrastive divergence
void update_hidden_bias_gradient(float * delta_c, float * p_0, float * p_k, int n);

// Update weights gradient
void update_weights_gradient(float ** delta_weigths, float * v_0, float * v, float * p_0, float * p_k, int n, int m);

// update the weigths in k-step contrastive divergence
//void update_weights(float ** delta_w, float * v_0, float * v_k, float * p_0, float * p_k);

// compute P(H_{i} = 1 | v)
// dim(wvc) = n x 1
void compute_probability_hidden_given_visible(float * wvc, float * p, int n);

// compute P(V_{j} = 1 | h)
// dim(whb) = m x 1
void compute_probability_visible_given_hidden(float * whb, float * p, int m);

// sample h_{i}^{t} ~ p(h_{i} | v^{t})
// or v_{j}^{t} ~ p(v_{j} | h^{t})
// i=1..n
// j=1..m
void sample_vector(float * vec, float * p, int n);

// compute hidden nodes given model parameters and for a new visible input
void compute_hidden(float ** weights, float * hidden_bias, float * visible, float * hidden, int num_hidden_units, int num_visible_units);


void generate_features_DL_Implementation(std::string data_file, const char * mdl_weight_file,
                const char * mdl_visible_bias_file, const char * mdl_hidden_bias_file,
                const char * learned_features_file, int number_of_examples, int num_visible_units, int num_hidden_units);
#endif // RBM_UTIL_H_INCLUDED
