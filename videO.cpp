#include <queue>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <SFML/Graphics.hpp>
#include <mutex>
#include <random>
#include "videO.h"

//alsa and audio stuff
int* freqs;
int* seconds;
char*** sineWaves;
char* samplebuffer;
snd_pcm_uframes_t frames;
snd_pcm_hw_params_t *params;
int size2;
uint val;

char* audiO::generateSineWaves(bool** note_matrix, int* freqs, int* seconds){ 
    char*** array = audiO::fill3DArrayWithSoundingSines(note_matrix, audiO::MATRIX_X_SIZE, audiO::MATRIX_Y_SIZE, freqs);
    int noteIndex = 0;
    for (int i=0; i < audiO::MATRIX_Y_SIZE; i++){
        for (int j=0; j < audiO::MATRIX_X_SIZE; j++){
            for (int k=0; k < audiO::SIZE; k++){
                audiO::alsabuffer[k] += array[i][j][k];
            }
        }
    }
    return audiO::alsabuffer;
}

int* audiO::generateFreqs(int size){
    int* array = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++){
        array[i] = 440*pow(2, (i - (audiO::NUM_SINES/2)) * 5.0f / 12.0f);
    }
    return array;
}

int* audiO::generateSeconds(int size){
    int* array = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++){
        array[i] = audiO::NUM_SECONDS;
    }
    return array;
}


bool** audiO::generateNoteMatrix(int xSize, int ySize){
    bool** array = (bool**)calloc(xSize, sizeof(bool*));
    for (int i = 0; i < xSize; i++){
        array[i] = (bool*)calloc(ySize, sizeof(bool));
        for (int j = 0; j < ySize; j++){
            array[i][j] = false;
            if (j%4 == 0){
                array[i][j] = true;
            }
        }
    }
    return array;
}

char*** audiO::fill3DArrayWithSoundingSines(bool** array, int xSize, int ySize, int* freqs){
    char*** array2 = (char***)calloc(xSize, sizeof(char**));
    for (int i = 0; i < xSize; i++){
        array2[i] = (char**)calloc(ySize, sizeof(char*));
        for (int j = 0; j < ySize; j++){
            array2[i][j] = (char*)calloc(audiO::SIZE, sizeof(char));
            if (array[i][j] == true){
                for (int k = 0; k < audiO::SIZE; k++){
                    array2[i][j][k] = (char) (sin((double)k/((double)audiO::SIZE/freqs[i*j])*M_PI*2.)*127+128);
                }
            }
            else{
                for (int k = 0; k < audiO::SIZE; k++){
                    array2[i][j][k] = 0;
                }
            }
        }
    }
    return array2;
}

void audiO::play_alsa(char* buffer, int size, snd_pcm_t *handle_alsa, int rc_alsa){
    rc_alsa = snd_pcm_writei(handle_alsa, buffer, size);
    if (rc_alsa == -EPIPE) {
        // EPIPE means underrun
        fprintf(stderr, "underrun occurred\n");
        snd_pcm_prepare(handle_alsa);
    } else if (rc_alsa < 0) {
        fprintf(stderr, "error from writei: %s\n", snd_strerror(rc_alsa));
    }  else if (rc_alsa != (int)size) {
        fprintf(stderr, "short write, write %d frames\n", rc_alsa);
    }
}

void audiO::play_alsa_thread(char* buffer){
    for (int i = 0; i < audiO::PLAY_LOOPS; i++){
        audiO::play_alsa(buffer, audiO::SIZE, audiO::handle_alsa, audiO::rc_alsa);
    }
}



void audiO::alsaSetup(){
    snd_pcm_hw_params_t *params;
    uint val;
    int dir;
    snd_pcm_uframes_t frames;
    
    // open pcm device
    int err = snd_pcm_open(&audiO::handle_alsa, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        printf("Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    
    // allocate a hardware parameters object
    snd_pcm_hw_params_alloca(&params);
    
    // fill it in with default values
    snd_pcm_hw_params_any(audiO::handle_alsa, params);
    
    // set the desired hardware parameters
    // INTERLEAVED is the default
    snd_pcm_hw_params_set_access(audiO::handle_alsa, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    
    // signed 16 bit little endian format
    snd_pcm_hw_params_set_format(audiO::handle_alsa, params, SND_PCM_FORMAT_S16_LE);
    
    // two channels (stereo)
    snd_pcm_hw_params_set_channels(audiO::handle_alsa, params, audiO::NUM_CHANNELS);
    
    // 44100 bits/second sampling rate (CD quality)
    val = audiO::SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(audiO::handle_alsa, params, &val, &dir);
    
    // set period size to 32 frames
    frames = audiO::NUM_FRAMES;
    
    // write the parameters to the driver
    audiO::rc_alsa = snd_pcm_hw_params(audiO::handle_alsa, params);
    if (audiO::rc_alsa < 0) {
        fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(audiO::rc_alsa));
        exit(1);
    }
}

void audio_thread(){
    while (audiO::running){
        audiO::play_alsa_thread(audiO::alsabuffer);
    }
}

void audiO::start_audio(){
    audiO::running = true;
    std::thread t0(audio_thread);
    t0.detach();
}

void audiO::stop_audio(){
    audiO::running = false;
}


void setupArrays() {
    int* freqs = (int*)calloc(audiO::MATRIX_ELEMENTS, sizeof(int));
    int* seconds = (int*)calloc(audiO::MATRIX_ELEMENTS, sizeof(int));
    char*** sineWaves = (char***)calloc(audiO::MATRIX_ELEMENTS, sizeof(char**));
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++) {
        sineWaves[i] = (char**)calloc(audiO::MATRIX_ELEMENTS, sizeof(char*));
        for (int j = 0; j < audiO::MATRIX_ELEMENTS; j++) {
            sineWaves[i][j] = (char*)calloc(audiO::SIZE, sizeof(char));
        }
    }
    char* samplebuffer = (char*)calloc(audiO::SIZE, sizeof(char));
}

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

//ALSA stuff
bool** translateFiringsToNoteMatrix(Network* network) {
    bool** noteMatrix = (bool**)calloc(audiO::MATRIX_ELEMENTS, sizeof(bool*));
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++) {
        noteMatrix[i] = (bool*)calloc(audiO::MATRIX_ELEMENTS, sizeof(bool));
    }
    int layerIndex = 0;
    int neuronIndex = 0;
    for (auto layer : network->layers) {
        layerIndex++;
        for (auto neuron : layer->neurons) {
            neuronIndex++;
            if (neuron->firing) {
                noteMatrix[layerIndex][neuronIndex] = true;
            }
        }
    }
    return noteMatrix;
}

//SMFL threaded display
std::mutex mutex;

void display(Network* network, sf::RenderWindow* window) {
    while (window->isOpen()) {
        mutex.lock();
        window->clear();
        network->update();
        audiO::alsabuffer = audiO::generateSineWaves(translateFiringsToNoteMatrix(network), freqs, seconds);            
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
        //mutex.unlock();
    }
}

int main() {
    Network network(10, 10);
    network.setWeights(0.6);
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Neural Network");
    std::thread display_thread(display, &network, &window);
    setupArrays();
    audiO::alsaSetup();
    audiO::start_audio();
    display_thread.join();
    audiO::stop_audio();
    return 0;
}




