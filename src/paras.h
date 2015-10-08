#ifndef PARAS_H_INCLUDED
#define PARAS_H_INCLUDED

struct rbm_paras{
    int num_visible_units;
    int num_hidden_units;
    int K;  // number of gibbs sampling steps in cd
    int num_epochs; // number of epochs of approximated gradient ascent
    float learning_rate;    // sgd learning rate
    int size_minibatch;
};

void load_rbm_paras(rbm_paras & para, const char * file_rbm_paras);

#endif // PARAS_H_INCLUDED
