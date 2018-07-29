#include <iostream>
#include <vector>
#include "nnet.h"
#include <cstdlib>
#include <cassert>
#include <cmath>

/*------------------------------------------ class Neuron --------------------------------*/
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

double Neuron::randomWeight(){
    return rand()/double(RAND_MAX);
}

void Neuron::feedforward(const Layer &prevLayer){
    double sum = 0.0;
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputval = Neuron::transferfunction(sum);
}

double Neuron::transferfunction(double x){
    // tangh function
    return tanh(x);
}

double Neuron::transferfunctionDerivative(double x){
    return 1.0 - x*x;
}

void Neuron::calcOutputGradients(double targetval){
    double delta = targetval - m_outputval;
    m_gradient = delta * Neuron::transferfunctionDerivative(m_outputval);
}

void Neuron::calcHiddenGradients(Layer &nextLayer){
    double dow = sumDow(nextLayer);
    m_gradient = dow*Neuron::transferfunctionDerivative(m_outputval);
}

double Neuron::sumDow(const Layer &nextLayer) const {
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInuputWeights(Layer &prevLayer){
    // weights are updated in the connection container
    for(unsigned n = 0; n < prevLayer.size() - 1; ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta*neuron.getOutputVal()*m_gradient + alpha*oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

/*------------------------------------------ class Net -----------------------------------*/
Net::~Net(){ }

Net::Net(const std::vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for (unsigned layernum = 0; layernum < numLayers; ++layernum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layernum == numLayers-1 ? 0:topology[layernum + 1];
        std::cout << "Added a layer." << std::endl;
        for (unsigned neuronnum = 0; neuronnum <= topology[layernum]; ++neuronnum){
            m_layers.back().push_back(Neuron(numOutputs, neuronnum));
            std::cout << "Neuron added" << std::endl;
        }
        
        // force bias value to be 1.0
        m_layers.back().back().setoutputval(1.0);
    }
}

void Net::feedforward(const std::vector<double> &train_x){
    assert(train_x.size() == m_layers[0].size() - 1);
    for (unsigned i = 0; i < train_x.size(); ++i){
        m_layers[0][i].setoutputval(train_x[i]);
    }

    // forward propagation
    for (unsigned layernum = 1; layernum < m_layers.size(); ++layernum){
        Layer &prevLayer = m_layers[layernum - 1];
        for (unsigned n = 0; n < m_layers[layernum].size() - 1; ++n){
            m_layers[layernum][n].feedforward(prevLayer);
        }
    }
}

void Net::backprop(const std::vector<double> &train_y){
    // calculating overall errors
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = train_y[n] - outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    // running average error
    m_recentAverageError = (m_recentAverageError*m_recentAverageSmoothingFactor + m_error)/(m_recentAverageSmoothingFactor + 1.0);
    // calculating output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(train_y[n]);
    }
    // calculating gradients of inner layer
    for(unsigned layernum = m_layers.size() - 2; layernum > 0; --layernum){
        Layer &hiddenlayer = m_layers[layernum];
        Layer &nextlayer = m_layers[layernum + 1];
        for(unsigned n = 0; n < hiddenlayer.size(); ++n){
            hiddenlayer[n].calcHiddenGradients(nextlayer);
        }
    }

    // update connection weights
    for(unsigned layernum = m_layers.size() - 1; layernum > 0; --layernum){
        Layer &layer = m_layers[layernum];
        Layer &prevLayer = m_layers[layernum - 1];

        for(unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInuputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &results) const {
    results.clear();
    for(unsigned n = 0; n < m_layers.back().size() -1 ; ++n){
        results.push_back(m_layers.back()[n].getOutputVal());
    }
}