#include <queue>
#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include "videO.h"

//audiO stuff

audiO::Oscillator::Oscillator(float freq, float amp, float phase){
    this->freq = freq;
    this->amp = amp;
    this->phase = phase;
    this->phase_increment = this->freq / audiO::SAMPLE_RATE;
}

float audiO::Oscillator::getSample(){
    float sample = this->amp * sin(2 * M_PI * this->phase);
    this->phase += this->phase_increment;
    if (this->phase > 1){
        this->phase -= 1;
    }
    return sample;
}

void audiO::Oscillator::setFrequency(float frequency){
    this->freq = frequency;
    this->phase_increment = this->freq / audiO::SAMPLE_RATE;
}

void audiO::Oscillator::setAmplitude(float amplitude){
    this->amp = amplitude;
}

audiO::OscillatorBank::OscillatorBank(){
    this->num_oscillators = audiO::MATRIX_ELEMENTS;
    for (int i = 0; i < this->num_oscillators; i++){
        this->oscillators[i] = new audiO::Oscillator(audiO::freqs[i], audiO::AMPLITUDE_NEURON, 0);
    }
}

float audiO::OscillatorBank::getSample(){
    float sample = 0;
    for (int i = 0; i < this->num_oscillators; i++){
        sample += this->oscillators[i]->getSample();
    }
    return sample;
}

float* audiO::generateSineWaves(){
    int note_index = 0;
    for (int i = 0; i < audiO::MATRIX_X_SIZE; i++){
        for (int j = 0; j < audiO::MATRIX_Y_SIZE; j++){
            if (audiO::note_matrix[i][j] == true){
                audiO::global_oscillator_bank->oscillators[note_index]->setAmplitude(audiO::AMPLITUDE_NEURON * videO::globalNetwork->layers[i]->neurons[j]->activation);
            }
            else{
                audiO::global_oscillator_bank->oscillators[note_index]->setAmplitude(0);
            }
            note_index++;
        }
    }
    for (int i = 0; i < audiO::NUM_FRAMES; i++){
        audiO::alsabuffer[i] = audiO::global_oscillator_bank->getSample();
    }
    //normalize signal
    //1. find max
    float max = 0;
    for (int i = 0; i < audiO::NUM_FRAMES; i++){
        if (audiO::alsabuffer[i] > max){
            max = audiO::alsabuffer[i];
        }
    }
    //2. divide all by max
    for (int i = 0; i < audiO::NUM_FRAMES; i++){
        audiO::alsabuffer[i] = (audiO::alsabuffer[i] / max) * audiO::SAMPLE_MAX;
    }

    return audiO::alsabuffer;
}
/*
float* audiO::generateSineWaves(){ 
    int note_index = 0;
    for (int i = 0; i < audiO::MATRIX_X_SIZE; i++){
        for (int j = 0; j < audiO::MATRIX_Y_SIZE; j++){
            note_index++;
            if (audiO::note_matrix[i][j] == true){
                for (int k = 0; k < audiO::NUM_FRAMES; k++){
                    audiO::sinewaves[i][j][k] = (float)(sin(2 * M_PI * audiO::freqs[note_index] / audiO::SAMPLE_RATE) * (audiO::AMPLITUDE_NEURON * videO::globalNetwork->layers[i]->neurons[j]->activation));
                }
            }
            else{
                for (int k = 0; k < audiO::NUM_FRAMES; k++){
                    audiO::sinewaves[i][j][k] = 0;
                }
            }
        }
    }

    for (int i=0; i < audiO::MATRIX_X_SIZE; i++){
        for (int j=0; j < audiO::MATRIX_Y_SIZE; j++){
            for (int k=0; k < audiO::NUM_FRAMES; k++){
                if (i == 0 && j == 0){
                    audiO::alsabuffer[k] = audiO::sinewaves[i][j][k];
                }
                else{
                    audiO::alsabuffer[k] += audiO::sinewaves[i][j][k];
                }
            }
        }
    }
    //normalize signal
    //1. find max
    float max = 0;
    for (int i = 0; i < audiO::NUM_FRAMES; i++){
        if (audiO::alsabuffer[i] > max){
            max = audiO::alsabuffer[i];
        }
    }
    //2. divide all by max
    for (int i = 0; i < audiO::NUM_FRAMES; i++){
        audiO::alsabuffer[i] = (audiO::alsabuffer[i] / max) * audiO::SAMPLE_MAX;
    }

    return audiO::alsabuffer;
}
*/

float* audiO::generateFreqs(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::freqs[i] = audiO::note_map[i];
    }
    return audiO::freqs;
}


float* audiO::generateSeconds(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::seconds[i] = audiO::NUM_SECONDS;
    }
    return audiO::seconds;
}

void audiO::generateNoteMap(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::note_map[i] = 220 * pow(2, (double)i / 12); //equal temperament A4 = 440Hz
    }
}


bool** audiO::generateNoteMatrix(){
    for (int i = 0; i < audiO::MATRIX_X_SIZE; i++){
        for (int j = 0; j < audiO::MATRIX_Y_SIZE; j++){
            audiO::note_matrix[i][j] = false;
        }
    }
    return audiO::note_matrix;
}
//converts the firing neurons to a matrix of booleans
bool** audiO::fireToBool(){
    int layerIndex = 0;
    int neuronIndex = 0;
    for (auto layer : videO::globalNetwork->layers) {
        layerIndex++;
        for (auto neuron : layer->neurons) {
            neuronIndex++;
            if (neuron->firing) {
                audiO::note_matrix[layerIndex][neuronIndex] = true;
            }
            else {
                audiO::note_matrix[layerIndex][neuronIndex] = false;
            }
        }
    }
    return audiO::note_matrix;
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
    err = snd_pcm_hw_params_any(audiO::handle_alsa, params);
    if (err < 0) {
        fprintf(stderr, "Can not configure this PCM device: %s\n", snd_strerror(err));
        exit(1);
    }
    
    // set the desired hardware parameters
    // INTERLEAVED is the default
    err = snd_pcm_hw_params_set_access(audiO::handle_alsa, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        fprintf(stderr, "Error setting interleaved mode: %s\n", snd_strerror(err));
        exit(1);
    }
    
    // signed 16 bit little endian format
    err = snd_pcm_hw_params_set_format(audiO::handle_alsa, params, SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        fprintf(stderr, "Error setting format: %s\n", snd_strerror(err));
        exit(1);
    }
    
    // two channels (stereo)
    err = snd_pcm_hw_params_set_channels(audiO::handle_alsa, params, audiO::NUM_CHANNELS);
    if (err < 0) {
        fprintf(stderr, "Error setting channels: %s\n", snd_strerror(err));
        exit(1);
    }
    
    // 44100 bits/second sampling rate (CD quality)
    val = audiO::SAMPLE_RATE;
    dir = 0;
    err = snd_pcm_hw_params_set_rate_near(audiO::handle_alsa, params, &val, &dir);
    if (err < 0) {
        fprintf(stderr, "Error setting sampling rate (%d): %s\n", val, snd_strerror(err));
        exit(1);
    }
    
    audiO::frames = audiO::NUM_FRAMES*sizeof(char);
    err = snd_pcm_hw_params_set_period_size_near(audiO::handle_alsa, params, &audiO::frames, &dir);
    if (err < 0) {
        fprintf(stderr, "Error setting period size (%d): %s\n", frames, snd_strerror(err));
        exit(1);
    }
    
    // write the parameters to the driver
    audiO::rc_alsa = snd_pcm_hw_params(audiO::handle_alsa, params);
    if (audiO::rc_alsa < 0) {
        fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(audiO::rc_alsa));
        exit(1);
    }
}

std::mutex mtx;
void audiO::audio_thread(){
    while (audiO::running){
        mtx.lock();
        audiO::note_matrix = audiO::fireToBool();
        audiO::alsabuffer = audiO::generateSineWaves();
        char* buffer = (char*)malloc(audiO::NUM_FRAMES * sizeof(char));
        for (int i = 0; i < audiO::NUM_FRAMES; i++){
            buffer[i] = (char)(audiO::alsabuffer[i]);
        }

        audiO::rc_alsa = snd_pcm_writei(audiO::handle_alsa, buffer, audiO::frames);
        if (audiO::rc_alsa == -EPIPE) {
            // EPIPE means underrun
            fprintf(stderr, "underrun occurred\n");
            snd_pcm_prepare(audiO::handle_alsa);
        } else if (audiO::rc_alsa < 0) {
            fprintf(stderr, "error from writei: %s\n", snd_strerror(audiO::rc_alsa));
        } else if (rc_alsa != (int)audiO::frames) {
            fprintf(stderr, "short write, write %d frames\n", audiO::rc_alsa);
        }
        free(buffer);
        mtx.unlock();
        //std::this_thread::sleep_for(std::chrono::milliseconds((int)(audiO::SAMPLE_RATE / audiO::NUM_FRAMES)));

    }
}

void audiO::start_audio(){
    audiO::running = true;
    std::thread t0(audiO::audio_thread);
    t0.detach();
}

void audiO::stop_audio(){
    audiO::running = false;
}

void audiO::setupArrays() {
    audiO::freqs = (float*)malloc(audiO::MATRIX_ELEMENTS * sizeof(float));
    audiO::seconds = (float*)malloc(audiO::MATRIX_ELEMENTS * sizeof(float));
    audiO::note_matrix = (bool**)malloc(audiO::MATRIX_X_SIZE * sizeof(bool*));
    for (int i = 0; i < audiO::MATRIX_X_SIZE; i++) {
        audiO::note_matrix[i] = (bool*)malloc(audiO::MATRIX_Y_SIZE * sizeof(bool));
    }
    audiO::alsabuffer = (float*)malloc(audiO::NUM_FRAMES * sizeof(float));
}

void audiO::freeArrays() {
    free(audiO::freqs);
    free(audiO::seconds);
    for (int i = 0; i < audiO::MATRIX_X_SIZE; i++) {
        free(audiO::note_matrix[i]);
    }
    free(audiO::note_matrix);
    free(audiO::alsabuffer);
    snd_pcm_drain(audiO::handle_alsa);
    snd_pcm_close(audiO::handle_alsa);
}

//neural network and video stuff

videO::Network* videO::Network::getNetwork() {
    return videO::globalNetwork;
}

double videO::sigmoid(double x) {
    //expexts x to be between 0 and 1 and xout will be between 0 and 1 too
    double xout = 1 / (1 + exp(-x))-0.5;
    double xoutmax = 1 / (1 + exp(-1.0))-0.5;

    xout/=xoutmax;
    return xout;
}

videO::Neuron::Neuron() {
    activation = 0;
    firing = false;
    connections = std::vector<videO::Connection*>();
    add_to_counter = 0.06;
}

videO::Connection::Connection(videO::Neuron* n1, videO::Neuron* n2, double w): neuron_from(n1), neuron_to(n2), weight(w) {
    n1->addConnection(this);
    n2->addConnection(this);
};

void videO::Connection::update() {
    double act = neuron_to->getActivation(neuron_from, weight);
}

videO::Layer::Layer(videO::Network* net) {
    network = net;
    neurons = std::vector<videO::Neuron*>();
    connections = std::vector<videO::Connection*>();
}

void videO::Layer::addNeuron(Neuron* neuron) {
    neurons.push_back(neuron);
}

void videO::Layer::addConnection(Connection* connection) {
    connections.push_back(connection);
}

void videO::Layer::update() {
    for (int i = 0; i < connections.size(); i++) {
        connections[i]->update();
    }

}

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


videO::Network::Network(int n_layers, int n_neurons) {
    for (int i = 0; i < n_layers; i++) {
        videO::Layer* layer = new videO::Layer(this);
        for (int j = 0; j < n_neurons; j++) {
            Neuron* neuron = new Neuron();
            layer->addNeuron(neuron);
        }
        addLayer(layer);
    }
    connect();
    target = 1;
}

void videO::Network::setTarget(int t) {
    target = t;
}

void videO::Network::addLayer(videO::Layer* layer) {
    layers.push_back(layer);
    layer->network = this;
}

void videO::Network::update() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update();
    }
    this->backpropWithTarget();
}

void videO::Network::backpropWithTarget() {
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
            deltas[i][j] = errors[i][j] * videO::sigmoid(activations[i][j]);
        }
    }

    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->neurons.size(); j++) {
            for (int k = 0; k < layers[i]->neurons[j]->connections.size(); k++) {
                layers[i]->neurons[j]->connections[k]->weight += videO::learning_rate * deltas[i][j] * layers[i]->neurons[j]->connections[k]->neuron_from->activation;
            }
        }
    }
}

    

void videO::Network::setWeights(double weight) {
    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->connections.size(); j++) {
            layers[i]->connections[j]->weight = weight;
        }
    }
}

void videO::Network::connectSelf(videO::Layer* layer) {
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

void videO::Network::connectLayers() {
    for (int i = 0; i < layers.size() - 1; i++) {
        connectWithLayer(layers[i], layers[i + 1]);
    }
}


void videO::Network::connectWithLayer(Layer* layer1, Layer* layer2) {
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

void videO::Network::connect() {
    for (int i = 0; i < layers.size(); i++) {
        connectSelf(layers[i]);
    }
    connectLayers();
}


void videO::Neuron::addConnection(Connection* connection) {
        connections.push_back(connection);
    }

double videO::Neuron::getActivation(Neuron* neuron, double weight) {

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

void videO::display(sf::RenderWindow* window) {
    while (window->isOpen()) {
        mutex.lock();
        window->clear();
        videO::globalNetwork->update();           
        for (int i = 0; i < videO::globalNetwork->layers.size(); i++) {
            for (int j = 0; j < videO::globalNetwork->layers[i]->neurons.size(); j++) {
                double radius = videO::globalNetwork->layers[i]->neurons[j]->activation * 32;
                sf::CircleShape circle(radius);
                circle.setPosition(100 * i+50-radius, 100 * j+50-radius);
                if (videO::globalNetwork->layers[i]->neurons[j]->firing) {
                    circle.setFillColor(sf::Color::Red);
                    videO::globalNetwork->layers[i]->neurons[j]->firing = false;
                }
                else {
                    circle.setFillColor(sf::Color::White);
                    videO::globalNetwork->layers[i]->neurons[j]->firing = false;
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
                    videO::globalNetwork->setWeights(0.6);
                }
                if (event.key.code == sf::Keyboard::Enter) {
                    videO::globalNetwork->setWeights(-0.01);
                }
                if (event.key.code == sf::Keyboard::Num1) {
                    videO::globalNetwork->setTarget(1);
                }
                if (event.key.code == sf::Keyboard::Num2) {
                    videO::globalNetwork->setTarget(2);
                }
                if (event.key.code == sf::Keyboard::Num3) {
                    videO::globalNetwork->setTarget(3);
                }
                if (event.key.code == sf::Keyboard::Num4) {
                    videO::globalNetwork->setTarget(4);
                }
            }
        }

        window->display();
        mutex.unlock();
    }
}

int main() {
    videO::globalNetwork = new videO::Network(videO::NUM_LAYERS, videO::NUM_NEURONS);
    videO::globalNetwork->setWeights(0.6);
    audiO::global_oscillator_bank = new audiO::OscillatorBank();
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Neural Network");
    
    audiO::setupArrays(); // allocate memory for arrays used by the sound synthesis
    //the arrays are; freqs, seconds, sinewaves, note_matrix, alsabuffer
    //freqs is an array of doubles that will be used to determine the frequency of each note. It is a 100 element array, one for each neuron.
    audiO::generateFreqs(); // equal temperament A4 = 440Hz
    audiO::generateSeconds(); // a constant for now, but could be used to change the length of the note
    audiO::generateNoteMatrix(); // a matrix of booleans that will be used to determine which notes to play.
    //the matrix is 10x10 buffer length, so there are 100 notes for a 100 neurons.
    //If true, the note will be played, if false, it will not.
    //alsabuffer is a buffer of chars that will be used to store the sound data.
    audiO::generateNoteMap(); // a map that will be used to map the note_matrix index to a frequency

    audiO::alsaSetup();
    std::thread display_thread(videO::display, &window);
    audiO::start_audio();
    display_thread.join();
    audiO::stop_audio();
    audiO::freeArrays();
    return 0;
}




