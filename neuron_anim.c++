#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <SFML/Graphics.hpp>
#include <mutex>
#include <random>

const double learning_rate = 0.001;
const double threshold = 0.5;
const int len_at = 2^10;

class Connection;

class Neuron {
public:
    double activation;
    std::vector<double> at;
    int at_index;
    bool firing;
    std::vector<Connection*> connections;

    Neuron();
    
    void addConnection(Connection* connection);

    double getActivation(Neuron* neuron, double weight);

    void fire() {
        firing = true;
    }
};

std::mutex connectionMutex;

class Connection {
public:
    Neuron* neuron1;
    Neuron* neuron2;
    double weight1;
    double weight2;
    bool running;

    Connection(Neuron* n1, Neuron* n2, double w1, double w2) : neuron1(n1), neuron2(n2), weight1(w1), weight2(w2), running(true) {}

    void update() {
            double act1 = neuron1->getActivation(neuron2, weight1);
            double act2 = neuron2->getActivation(neuron1, weight2);
            double delta = act1 - act2;
            updateWeights(delta);
    }                                                           

    void updateWeights(double delta) {
        weight1 += learning_rate * delta * neuron2->activation;
        weight2 += learning_rate * delta * neuron1->activation;
    }
};

Neuron::Neuron() {
        activation = 0;
        // Initialize at with sigmoid function
        at.resize(len_at);
        for (int i = 0; i < len_at; ++i) {
            at[i] = (1 / (1 + std::exp(-static_cast<double>(i) / len_at))) - 0.5;
        }

        // Normalize the values
        double maxAt = *std::max_element(at.begin(), at.end());
        for (int i = 0; i < len_at; ++i) {
            at[i] *= 1 / maxAt;
        }
        std::cout << at[0] << std::endl;
        std::cout << at[len_at-1] << std::endl;
        at_index = 0;
        firing = false;
        connections = std::vector<Connection*>();
    }

void Neuron::addConnection(Connection* connection) {
        connections.push_back(connection);
    }

double Neuron::getActivation(Neuron* neuron, double weight) {
        // calculate activation
        activation = at[at_index];
        at_index = (at_index + 1) % len_at;
        for (auto connection : connections) {
            if (connection->neuron1 == neuron) {
                activation += connection->weight1 * connection->neuron2->activation;
            }
            else {
                activation += connection->weight2 * connection->neuron1->activation;
            }
        }

        return activation;
    }


class NeuronLayer {
public:
    std::vector<Neuron*> neurons;
    std::vector<Connection*> connections;

    NeuronLayer(int n_neurons) {
        for (int i = 0; i < n_neurons; ++i) {
            neurons.push_back(new Neuron());
        }
    }

    void update() {
        for (auto connection : connections) {
            connection->update();
        }
    }

    void addConnection(Neuron* neuron1, Neuron* neuron2, double weight1, double weight2) {
        Connection* connection = new Connection(neuron1, neuron2, weight1, weight2);
        connections.push_back(connection);
        neuron1->addConnection(connection);
        neuron2->addConnection(connection);
    }

    void setWeights(double weight1, double weight2) {
        for (auto connection : connections) {
            connection->weight1 = weight1;
            connection->weight2 = weight2;
        }
    }

    void connectWithLayer(NeuronLayer* layer) {
        for (auto neuron1 : neurons) {
            for (auto neuron2 : layer->neurons) {
                addConnection(neuron1, neuron2, 0, 0);
            }
        }
    }

    void connectSelf() {
        for (auto neuron1 : neurons) {
            for (auto neuron2 : neurons) {
                addConnection(neuron1, neuron2, 0, 0);
            }
        }
    }

    std::vector<double> getActivations() {
        std::vector<double> activations;
        for (auto neuron : neurons) {
            activations.push_back(neuron->activation);
        }
        return activations;
    }
};

class NeuralNetwork {
public:
    std::vector<NeuronLayer*> layers;
    int n_layers;
    int n_neurons_per_layer;

    NeuralNetwork(int n_layers, int n_neurons_per_layer) {
        this->n_layers = n_layers;
        this->n_neurons_per_layer = n_neurons_per_layer;
        for (int i = 0; i < n_layers; ++i) {
            layers.push_back(new NeuronLayer(n_neurons_per_layer));
        }
    }

    void update() {
        for (auto layer : layers) {
            layer->update();
        }
    }

    void setWeights(double weight1, double weight2) {
        for (auto layer : layers) {
            layer->setWeights(weight1, weight2);
        }
    }

    void getActivations() {
        std::vector<std::vector<double>> activations;
        for (auto layer : layers) {
            activations.push_back(layer->getActivations());
        }
    }

    void connectLayers() {
        for (int i = 0; i < layers.size() - 1; ++i) {
            layers[i]->connectWithLayer(layers[i + 1]);
        }
    }

    void connectSelf() {
        for (auto layer : layers) {
            layer->connectSelf();
        }
    }
};

std::mutex drawingMutex; // Mutex to protect shared data

void renderThread(sf::RenderWindow& window, NeuralNetwork& network) {
    while (window.isOpen()) {
        // Lock the mutex before accessing shared data
        std::lock_guard<std::mutex> lock(drawingMutex);

        window.clear(sf::Color::White);

        network.update();

        // Draw your neural network
        for (int i = 0; i < network.n_layers; ++i) {
            for (int j = 0; j < network.n_neurons_per_layer; ++j) {
                int radius = (int) (network.layers[i]->neurons[j]->activation * 100);
                sf::CircleShape circle(radius);
                circle.setPosition(100 * j, 100 * i);
                circle.setFillColor(sf::Color::Black);
                window.draw(circle);
            }
        }

        window.display();
    }
}

int main() {
    // Your existing code here

    // Create an SFML window
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Neural Network Visualization");

    // Start the rendering thread
    NeuralNetwork network(3, 5);
    std::thread renderingThread(renderThread, std::ref(window), std::ref(network));

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
    }

    // Wait for the rendering thread to finish
    renderingThread.join();

    return 0;
}

