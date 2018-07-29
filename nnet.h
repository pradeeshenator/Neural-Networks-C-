#pragma once
#include <vector>

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;
/*---------------------------------------- class Neuron ----------------------------------*/
class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setoutputval(double val) { m_outputval = val; };
    double getOutputVal(void) const { return m_outputval; };
    void feedforward(const Layer &prevLayer);
    void calcOutputGradients(double targetval);
    void calcHiddenGradients(Layer &nextLayer);
    void updateInuputWeights(Layer &prevLayer);
private:
    static double eta, alpha;
    double m_outputval;
    std::vector<Connection> m_outputWeights;
    static double randomWeight(void);
    unsigned m_myIndex;
    static double transferfunction(double x);
    static double transferfunctionDerivative(double x);
    double m_gradient;
    double sumDow(const Layer &nextLayer) const;
};

/*------------------------------------------ class Net -----------------------------------*/
class Net{
public:
    Net(const std::vector<unsigned> &topology);
    void feedforward(const std::vector<double> &train_x);
    void backprop(const std::vector<double> &train_y);
    void getResults(std::vector<double> &test_x) const;
    ~Net();

private:
    std::vector<Layer> m_layers; // m_layers[layer number][neuron number]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};