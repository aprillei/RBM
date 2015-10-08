#include "util.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>

// logistic function
float logistic(float x){
    return 1/(1+exp(-x));
}

//component-wise activation using logistic function
void activate_logistic(float * x, int dim){
	for(int i=0; i<dim; i++)
		x[i] = logistic(x[i]);
}

// TODO implement faster mult
// multiply matrix W to vector v
// W of dimension n x m
// v of dimension m x 1
void multiply_matrix_vector(float ** W, float * v, float * wv, int n, int m){
    for (int i=0; i<n; i++){
        float sum = 0;
        for (int j=0; j<m; j++){
            sum += W[i][j] * v[j];
        }
        wv[i] = sum;
    }
}

// TODO implement faster mult
// multiply matrix W to vector h column wise
// W of dimension n x m
// h of dimension n x 1
void multiply_matrix_vector_column_wise(float ** W, float * h, float * wh, int n, int m){
    for (int i=0; i<m; i++){
        float sum = 0;
        for (int j=0; j<n; j++){
            sum += W[j][i] * h[j];
        }
        wh[i] = sum;
    }
}

// add vector
// a = a + b
// a and b of size dim
void add_vector(float * a, float * b, int dim){
    for (int i=0; i<dim; i++){
        a[i] += b[i];
    }
}

// random float
float randfloat(){
	return ((rand()%10001) * 2 - 10000)/10000.0;
}

//copy vector b to vector a
void copy_vec(float * a, float * b, int dim){
	for(int i=0; i<dim; i++)
		a[i]=b[i];
}

// multiply vector a of dim dim by constant lambda
void multiply_constant_vector(float * a, float lambda, int dim){
    for (int i=0; i<dim; i++){
        a[i] *= lambda;
    }
}

// multiply matrix w if dimensions n x m by constant lambda
void multiply_constant_matrix(float ** a, float lambda, int n, int m){
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            a[i][j] *= lambda;
        }
    }
}

// add matrix b to matrix a
// dim(a) = n x m
void add_matrix(float ** a, float ** b, int n,int m){
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++){
            a[i][j] += b[i][j];
        }
    }
}

// Randomly initialize a vector of ints < max_int
void rand_init_vec_int(int * a, int dim, int max_int){
	for(int i=0;i<dim;i++){
        a[i]=rand()%max_int;
    }
}


// Convert the input txt file (just after conversion from .mat)
// into libsvm format
// so as to compare resuls with visible units to the onces with new learned features
void transform_to_libsvm_format(std::string data_file, int sample_size, int num_visible_units, std::string libsvm_features_file){
    std::string feature_file(data_file);
	std::string label_file(data_file);
	feature_file += ".fea";
	label_file += ".labels";

    float ** input_f = new float*[sample_size];
    for (int i=0; i<sample_size; i++){
        input_f[i] = new float[num_visible_units];
    }

    std::ifstream feat_file(feature_file.c_str(),std::ios::in);
    for (int i=0; i<sample_size; i++){
        for (int j=0; j<num_visible_units; j++){
            feat_file >> input_f[i][j];
            // Binary RBM
            if (input_f[i][j] > 0){
                input_f[i][j] = 1;
            }
        }
    }
    feat_file.close();

    int * output_labels = new int[sample_size];
    FILE * tmpfp = fopen(label_file.c_str(),"r");
	for (int i = 0; i < sample_size; i++){
		fscanf(tmpfp, "%d", &output_labels[i]);
	}
	fclose(tmpfp);

    // write the features to libsvm_features_file
    // libsvm uses a sparse format
    std::ofstream out_file(libsvm_features_file);
    if (out_file.is_open()){
        for (int i=0; i<sample_size; i++){
            int non_zero_features = 0;
            for (int j=0; j<num_visible_units; j++){
                if (input_f[i][j] != 0){
                    non_zero_features++;
                    if (non_zero_features == 1){
                        out_file << output_labels[i];
                    }
                    out_file << " " << (j+1) << ":" << input_f[i][j];   // indexing starts at 1 in libsvm
                }
            }
            if (non_zero_features > 0){
                out_file << std::endl;
            }
        }
        out_file.close();
    }
}
