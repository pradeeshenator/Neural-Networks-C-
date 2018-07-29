#include <iostream>
#include <vector>
#include "nnet.h"

int main(){

    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    std::vector<double> trainXvals;
    myNet.feedforward(trainXvals);

    std::vector<double> trainYvals;
    myNet.backprop(trainYvals);

    std::vector<double> results;
    myNet.getResults(results);

    return 0;
}