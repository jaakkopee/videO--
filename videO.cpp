#include <queue>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <SFML/Graphics.hpp>
#include <mutex>
#include <random>

const double learning_rate = 0.01;
const double globalThreshold = 0.9999;
const int len_at = 2^10;

class Connection;
class Neuron;
class Layer;
class Network;

double sigmoid(double x) {
    //expexts x to be between 0 and 1 and xout will be between 0 and 1 too
    double xout = 1 / (1 + exp(-x))-0.5;
    double xoutmax = 1 / (1 + exp(-1.0))-0.5;

    xout/=xoutmax;
    return xout;
}

class Neuron {
public:
    double activation;
    std::queue<double> at;
    bool firing;
    std::vector<Connection*> connections;
    double add_to_counter;

    Neuron();

    void addConnection(Connection* connection);

    double getActivation(Neuron* neuron, double weight);

    void init_at() {
        for (int i = 0; i < len_at; i++) {
            at.push(0);
        }
    }
};


class Connection {
public:
    Neuron* neuron_from;
    Neuron* neuron_to;
    double weight;

    Connection(Neuron* n1, Neuron* n2, double w);

    void update();
};

Connection::Connection(Neuron* n1, Neuron* n2, double w): neuron_from(n1), neuron_to(n2), weight(w) {
    n1->addConnection(this);
    n2->addConnection(this);
};

void Connection::update() {
    double act = neuron_to->getActivation(neuron_from, weight);
}

class Layer {
public:
    std::vector<Neuron*> neurons;
    std::vector<Connection*> connections;
    Network* network;

    Layer(Network* net) : network(net) {}

    void addNeuron(Neuron* neuron) {
        neurons.push_back(neuron);
    }

    void addConnection(Connection* connection) {
        connections.push_back(connection);
    }

    void update() {
        for (int i = 0; i < connections.size(); i++) {
            connections[i]->update();
        }

    }
};

//target patterns, squares inside squares
double target1[10][10] = {
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,1,1,0,0,0},
    {0,0,0,1,0,0,1,0,0,0},
    {0,1,0,1,0,0,1,0,1,0},
    {0,1,0,1,1,1,1,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0}
};

double target2[10][10] = {
    {0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,1,1,1,1,0,1,0},
    {0,1,0,1,0,0,1,0,1,0},
    {0,0,0,1,0,0,1,0,0,0},
    {0,0,0,1,1,1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0}
};

double target3[10][10] = {
    {0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0}
};

double target4[10][10] = {
    {0,0,0,0,0,0,0,0,0,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,0,1,1,1,1,0,1,0},
    {0,1,0,1,0,0,1,0,1,0},
    {0,1,0,1,0,0,1,0,1,0},
    {0,1,0,1,1,1,1,0,1,0},
    {0,1,0,0,0,0,0,0,1,0},
    {0,1,1,1,1,1,1,1,1,0},
    {0,0,0,0,0,0,0,0,0,0}
};

class Network {
public:
    std::vector<Layer*> layers;
    int target;

    Network(int n_layers, int n_neurons) {
        for (int i = 0; i < n_layers; i++) {
            Layer* layer = new Layer(this);
            for (int j = 0; j < n_neurons; j++) {
                Neuron* neuron = new Neuron();
                layer->addNeuron(neuron);
            }
            addLayer(layer);
        }
        connect();
        target = 1;
    }

    void setTarget(int t) {
        target = t;
    }

    void addLayer(Layer* layer) {
        layers.push_back(layer);
        layer->network = this;
    }

    void update() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->update();
        }
        this->backpropWithTarget();
    }

    void backpropWithTarget() {
        double** targets = new double*[layers.size()];
        for (int i = 0; i < layers.size(); i++) {
            targets[i] = new double[layers[i]->neurons.size()];
        }

        // copy target to targets
        if (target == 1) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i]->neurons.size(); j++) {
                    targets[i][j] = target1[i][j];
                }
            }
        }
        else if (target == 2) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i]->neurons.size(); j++) {
                    targets[i][j] = target2[i][j];
                }
            }
        }
        else if (target == 3) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i]->neurons.size(); j++) {
                    targets[i][j] = target3[i][j];
                }
            }
        }

        else if (target == 4) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i]->neurons.size(); j++) {
                    targets[i][j] = target4[i][j];
                }
            }
        }

        double** activations = new double*[layers.size()];
        for (int i = 0; i < layers.size(); i++) {
            activations[i] = new double[layers[i]->neurons.size()];
        }

        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                activations[i][j] = layers[i]->neurons[j]->activation;
            }
        }

        double** errors = new double*[layers.size()];
        for (int i = 0; i < layers.size(); i++) {
            errors[i] = new double[layers[i]->neurons.size()];
        }

        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                errors[i][j] = targets[i][j] - activations[i][j];
            }
        }

        double** deltas = new double*[layers.size()];
        for (int i = 0; i < layers.size(); i++) {
            deltas[i] = new double[layers[i]->neurons.size()];
        }

        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                deltas[i][j] = errors[i][j] * sigmoid(activations[i][j]);
            }
        }

        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                for (int k = 0; k < layers[i]->neurons[j]->connections.size(); k++) {
                    layers[i]->neurons[j]->connections[k]->weight += learning_rate * deltas[i][j] * layers[i]->neurons[j]->connections[k]->neuron_from->activation;
                }
            }
        }
    }

    

    void setWeights(double weight) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->connections.size(); j++) {
                layers[i]->connections[j]->weight = weight;
            }
        }
    }

    void connectSelf(Layer* layer) {
        for (int i = 0; i < layer->neurons.size(); i++) {
            for (int j = 0; j < layer->neurons.size(); j++) {
                Connection* connection1 = new Connection(layer->neurons[i], layer->neurons[j], 0);
                Connection* connection2 = new Connection(layer->neurons[j], layer->neurons[i], 0);
                layer->neurons[i]->addConnection(connection1);
                layer->neurons[j]->addConnection(connection1);
                layer->neurons[i]->addConnection(connection2);
                layer->neurons[j]->addConnection(connection2);
                layer->addConnection(connection1);
                layer->addConnection(connection2);
            }
        }
    }

    void connectLayers() {
        for (int i = 0; i < layers.size() - 1; i++) {
            connectWithLayer(layers[i], layers[i + 1]);
        }
    }


    void connectWithLayer(Layer* layer1, Layer* layer2) {
        for (int i = 0; i < layer1->neurons.size(); i++) {
            for (int j = 0; j < layer2->neurons.size(); j++) {
                Connection* connection1 = new Connection(layer1->neurons[i], layer2->neurons[j], 0.01);
                Connection* connection2 = new Connection(layer2->neurons[j], layer1->neurons[i], 0.01);
                layer1->neurons[i]->addConnection(connection1);
                layer2->neurons[j]->addConnection(connection1);
                layer1->neurons[i]->addConnection(connection2);
                layer2->neurons[j]->addConnection(connection2);
                layer1->addConnection(connection1);
                layer2->addConnection(connection1);
                layer1->addConnection(connection2);
                layer2->addConnection(connection2);
            }
        }
    }

    void connect() {
        for (int i = 0; i < layers.size(); i++) {
            connectSelf(layers[i]);
        }
        connectLayers();
    }

};

Neuron::Neuron() {
        activation = 0;
        firing = false;
        connections = std::vector<Connection*>();
        add_to_counter = 0.06;
    }

void Neuron::addConnection(Connection* connection) {
        connections.push_back(connection);
    }

double Neuron::getActivation(Neuron* neuron, double weight) {

        this->activation += this->add_to_counter;
        for (auto connection : connections) {
            if (connection->neuron_to == this) {
                this->activation += connection->weight * connection->neuron_from->activation;
            }
        }

        if (this->activation < 0) {
            this->activation = 0;
        }
        if (this->activation > 1) {
            this->activation = 1;
        }

        this->activation = sigmoid(this->activation);


        if (this->activation > globalThreshold) {
            this->activation = 0;
            this->firing = true;
        }

        //this->at.push(this->activation);
        //this->at.pop();

        return activation;
    
    }

//SMFL threaded display
std::mutex mutex;

void display(Network* network, sf::RenderWindow* window) {
    while (window->isOpen()) {
        mutex.lock();
        window->clear();
        network->update();
        for (int i = 0; i < network->layers.size(); i++) {
            for (int j = 0; j < network->layers[i]->neurons.size(); j++) {
                double radius = network->layers[i]->neurons[j]->activation * 32;
                sf::CircleShape circle(radius);
                circle.setPosition(100 * i+50-radius, 100 * j+50-radius);
                if (network->layers[i]->neurons[j]->firing) {
                    circle.setFillColor(sf::Color::Red);
                    network->layers[i]->neurons[j]->firing = false;
                }
                else {
                    circle.setFillColor(sf::Color::White);
                    network->layers[i]->neurons[j]->firing = false;
                }
                window->draw(circle);
            }
        }
        //poll for events
        sf::Event event;
        while (window->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window->close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    network->setWeights(0.6);
                }
                if (event.key.code == sf::Keyboard::Enter) {
                    network->setWeights(-0.01);
                }
                if (event.key.code == sf::Keyboard::Num1) {
                    network->setTarget(1);
                }
                if (event.key.code == sf::Keyboard::Num2) {
                    network->setTarget(2);
                }
                if (event.key.code == sf::Keyboard::Num3) {
                    network->setTarget(3);
                }
                if (event.key.code == sf::Keyboard::Num4) {
                    network->setTarget(4);
                }
            }
        }

        window->display();
        mutex.unlock();
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Neural Network");
    Network network(10, 10);
    network.setWeights(0.6);

    std::thread display_thread(display, &network, &window);
    display_thread.join();
    return 0;
}




