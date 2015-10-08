#include "paras.h"
#include <fstream>
#include <string>
#include <iostream>

void load_rbm_paras(rbm_paras & para, const char * file_rbm_paras){
    std::string tmp;
    std::ifstream infile;

    infile.open(file_rbm_paras);
    infile>>tmp;
    para.num_visible_units = std::stoi(tmp);

    infile>>tmp;
    para.num_hidden_units = std::stoi(tmp);

    infile>>tmp;
    para.K = std::stoi(tmp);

    infile>>tmp;
    para.num_epochs= std::stoi(tmp);

    infile>>tmp;
    para.learning_rate = std::stof(tmp);

    infile>>tmp;
    para.size_minibatch = std::stoi(tmp);

    infile.close();
}
