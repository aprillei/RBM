#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

# include <cmath>
# include <string>
// logistic function
float logistic(float x);

//component-wise activation using logistic function
void activate_logistic(float * x, int dim);

// multiply a matrix to a vector
void multiply_matrix_vector(float ** W, float * v, float * wv, int n, int m);

// multiply a matrix to a vector column wise
void multiply_matrix_vector_column_wise(float ** W, float * h, float * wh, int n, int m);

// add vector
void add_vector(float * a, float * b, int dim);

// random float
float randfloat();

//copy vector b to vector a
void copy_vec(float * a, float * b, int dim);

// multiply vector a of dim dim by constant lambda
void multiply_constant_vector(float * a, float lambda, int dim);

// multiply matrix w if dimensions n x m by constant lambda
void multiply_constant_matrix(float ** w, float lambda, int n, int m);

// add matrix b to matrix a
// dim(a) = n x m
void add_matrix(float ** a, float ** b, int n,int m);

// Randomly initialize a vector of ints < max_int
void rand_init_vec_int(int * a, int dim, int max_int);

// Convert the input txt file (just after conversion from .mat)
// into libsvm format
// so as to compare resuls with visible units to the onces with new learned features
void transform_to_libsvm_format(std::string data_file, int sample_size, int num_visible_units, std::string libsvm_features_file);

#endif // UTIL_H_INCLUDED
