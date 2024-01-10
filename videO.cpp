#include <queue>
#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <vector>
#include "videO.h"

//audiO stuff

audiO::Oscillator::Oscillator(float freq, float amp, float phase){
    this->freq = freq;
    this->amp = amp;
    this->phase = phase;
    this->phase_increment = (this->freq / audiO::SAMPLE_RATE) * 2 * M_PI;
}

audiO::Oscillator::~Oscillator(){}

float audiO::Oscillator::getSample(){
    float sample = sin(this->phase) * (this->amp * this->freq)/1000;
    this->phase += this->phase_increment;
    if (this->phase > 2 * M_PI){
        this->phase -= 2 * M_PI;
    }
    return sample;
}

audiO::OscillatorBank::OscillatorBank(){
    this->num_oscillators = audiO::MATRIX_ELEMENTS;
    this->oscillators = new audiO::Oscillator*[this->num_oscillators];
    for (int i = 0; i < this->num_oscillators; i++){
        this->oscillators[i] = new audiO::Oscillator(audiO::freqs[i], 0.0, 0);
    }
}

audiO::OscillatorBank::~OscillatorBank(){
    for (int i = 0; i < this->num_oscillators; i++){
        delete this->oscillators[i];  // Delete each Oscillator object
    }
    delete[] this->oscillators;  // Delete the oscillators array
}

float audiO::OscillatorBank::getSample(){
    float sample = 0;
    for (int i = 0; i < this->num_oscillators; i++){
        sample += this->oscillators[i]->getSample();
    }
    return sample;
}

audiO::Synth::Synth(){
    this->oscillator_bank = new audiO::OscillatorBank();
    this->running = false;
    this->notes_playing = new bool[audiO::MATRIX_ELEMENTS];
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        this->notes_playing[i] = false;
    }
}

audiO::Synth::~Synth(){
    delete this->oscillator_bank;
    delete[] this->notes_playing;
}

void audiO::Synth::playNote(int note){
    this->notes_playing[note] = true;
}

void audiO::Synth::stopNote(int note){
    this->notes_playing[note] = false;
}

void audiO::Synth::process(float* output, int numFrames){
    for (int i = 0; i < numFrames; i++){
        float sample = 0;
        for (int j = 0; j < audiO::MATRIX_ELEMENTS; j++){
            Oscillator* osc = this->oscillator_bank->oscillators[j];
            osc->amp = 1;
            float osc_sample = osc->getSample();
            osc->amp = 0.0;
            if (this->notes_playing[j]){
                sample += osc_sample;
            }
        }
        output[i] = sample;
    }

}

float audiO::sigmoidSaturator(float x){
    //sigmoid function, value in in range -1 to 1, value out is in range -1 to 1
    float xout = (1 / (1 + exp(-x))-0.5)*2;
    return xout;
}


std::vector<float> audiO::generateFreqs(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::freqs[i] = audiO::note_map[i];
    }
    return audiO::freqs;
}


std::vector<float> audiO::generateSeconds(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::seconds[i] = audiO::NUM_SECONDS;
    }
    return audiO::seconds;
}

void audiO::generateNoteMap(){
    for (int i = 0; i < audiO::MATRIX_ELEMENTS; i++){
        audiO::note_map[i] = (i+1) * 43.2;
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
    
    audiO::frames = audiO::NUM_FRAMES;
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
        //videO::globalNetwork->update();
        float *buffer = new float[audiO::NUM_FRAMES];
        audiO::global_synth->process(buffer, audiO::NUM_FRAMES);
        for (int i = 0; i < audiO::NUM_FRAMES; i++){
            audiO::audiobuffer[i] = (short)(buffer[i] * audiO::SAMPLE_MAX);
        }
        audiO::rc_alsa = snd_pcm_writei(audiO::handle_alsa, audiO::audiobuffer, audiO::frames);
        if (audiO::rc_alsa == -EPIPE) {
            /* EPIPE means underrun */
            fprintf(stderr, "underrun occurred\n");
            snd_pcm_prepare(audiO::handle_alsa);
        }
        else if (audiO::rc_alsa < 0) {
            fprintf(stderr, "error from writei: %s\n", snd_strerror(audiO::rc_alsa));
        }
        else if (audiO::rc_alsa != (int)audiO::frames) {
            fprintf(stderr, "short write, write %d frames\n", audiO::rc_alsa);
        }
        mtx.unlock();
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
    audiO::freqs = std::vector<float>(audiO::MATRIX_ELEMENTS);
    audiO::seconds = std::vector<float>(audiO::MATRIX_ELEMENTS);
    audiO::audiobuffer = (short*)malloc(audiO::NUM_FRAMES * sizeof(short));
}


void audiO::freeArrays() {
    free(audiO::audiobuffer);
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
    add_to_counter = 0.001;
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

double target5[10][10] = {
{1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1},
{0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1.0},
{0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,1.0,0.9},
{0.7,0.6,0.5,0.4,0.3,0.2,0.1,1.0,0.9,0.8},
{0.6,0.5,0.4,0.3,0.2,0.1,1.0,0.9,0.8,0.7},
{0.5,0.4,0.3,0.2,0.1,1.0,0.9,0.8,0.7,0.6},
{0.4,0.3,0.2,0.1,1.0,0.9,0.8,0.7,0.6,0.5},
{0.3,0.2,0.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4},
{0.2,0.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3},
{0.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2}
};

//empty target
double target6[10][10] = {
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0}
};

//target with values 0.5
double target7[10][10] = {
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5},
    {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5}
};

//simple bassline
double target8[10][10] = {
    {1,0,0,0,0,0,0,0,0,0},
    {1,1.5,0,0,0,0,0,0,0,0},
    {1,0,1.5,0,0,0,0,0,0,0},
    {1,0,0,1.5,0,0,0,0,0,0},
    {1,0,0,0,1.5,0,0,0,0,0},
    {1,0,0,0,0,1.5,0,0,0,0},
    {1,0,0,0,0,0,1.5,0,0,0},
    {1,0,0,0,0,0,0,1.5,0,0},
    {1,0,0,0,0,0,0,0,1.5,0},
    {1,0,0,0,0,0,0,0,0,1.5}
};

//another simple bassline
double target9[10][10] = {
    {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5},
    {1,1.5,0,0,0,0,0,0,0,0},
    {1,0,1.5,0,0,0,0,0,0,0},
    {1,0,0,1.5,0,0,0,0,0,0},
    {1,0,0,0,1.5,0,0,0,0,0},
    {1,0,0,0,0,1.5,0,0,0,0},
    {1,0,0,0,0,0,1.5,0,0,0},
    {1,0,0,0,0,0,0,1.5,0,0},
    {1,0,0,0,0,0,0,0,1.5,0},
    {1,0,0,0,0,0,0,0,0,1.5}
};

//better bassline
double target0[10][10] = {
    {1.5,1.3,1.1,0.9,0.7,0.5,0.3,0.1,0.1,0.1},
    {1.4,1.2,1.0,0.8,0.6,0.4,0.2,0.0,0.0,0.0},
    {1.3,1.1,0.9,0.7,0.5,0.3,0.1,0.1,0.1,0.1},
    {1.2,1.0,0.8,0.6,0.4,0.2,0.0,0.0,0.0,0.0},
    {1.1,0.9,0.7,0.5,0.3,0.1,0.1,0.1,0.1,0.1},
    {1.0,0.8,0.6,0.4,0.2,0.0,0.0,0.0,0.0,0.0},
    {0.9,0.7,0.5,0.3,0.1,0.1,0.1,0.1,0.1,0.1},
    {0.8,0.6,0.4,0.2,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.7,0.5,0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
    {0.6,0.4,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
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

    else if (target == 5) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target5[i][j];
            }
        }
    }

    else if (target == 6) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target6[i][j];
            }
        }
    }

    else if (target == 7) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target7[i][j];
            }
        }
    }

    else if (target == 8) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target8[i][j];
            }
        }
    }

    else if (target == 9) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target9[i][j];
            }
        }
    }

    else if (target == 0) {
        for (int i = 0; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]->neurons.size(); j++) {
                targets[i][j] = target0[i][j];
            }
        }
    }

    int oscIndex = 0;
    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->neurons.size(); j++) {
            targets[i][j] += audiO::global_synth->oscillator_bank->oscillators[oscIndex]->amp;
            oscIndex++;
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

        if (this->activation > videO::globalThreshold && !this->firing && videO::firingNeurons.size() < videO::MAX_FIRING_NEURONS) {
            videO::firingNeurons.push_back(this);
            std::thread t0(videO::fireThread, this);
            t0.detach();
        }

        //this->at.push(this->activation);
        //this->at.pop();

        return activation;
    
    }

std::mutex fire_mtx;
void videO::fireThread(videO::Neuron* neuron) {
    fire_mtx.lock();
    //find out neurons index in the network
    int layerIndex = 0;
    int neuronIndex = 0;
    for (auto layer : videO::globalNetwork->layers) {
        for (auto n : layer->neurons) {
            if (n == neuron) {
                goto found;
            }
            neuronIndex++;
        }
        layerIndex++;
    }
    found:
    neuron->firing = true;
    neuron->activation = 0;
    int note_index = neuronIndex;
    audiO::global_synth->playNote(note_index);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    neuron->firing = false;
    audiO::global_synth->stopNote(note_index);
    videO::firingNeurons.erase(std::remove(videO::firingNeurons.begin(), videO::firingNeurons.end(), neuron), videO::firingNeurons.end());
    fire_mtx.unlock();
}

audiO::DelayLine::DelayLine(int buffer_size, float feedback, float wet, float dry){
    this->buffer_size = buffer_size;
    this->feedback = feedback;
    this->wet = wet;
    this->dry = dry;
    this->buffer = std::queue<float>();
    for (int i = 0; i < buffer_size; i++){
        this->buffer.push(0);
    }
}

audiO::DelayLine::~DelayLine(){
    while (!buffer.empty()){
        buffer.pop();
    }
}

void audiO::DelayLine::process(float* input, float* output, int numFrames){
    for (int i = 0; i < numFrames; i++){
        float in = input[i];
        float out = buffer.front();
        buffer.pop();
        buffer.push(in + out * feedback);
        output[i] = in * dry + out * wet;
    }
}





std::mutex neuron_mtx;
void videO::neuronThread() {
    while (videO::nt_running) {
        neuron_mtx.lock();
        videO::globalNetwork->update();  
        neuron_mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

//SMFL threaded display
std::mutex mutex;

void videO::display(sf::RenderWindow* window) {
    while (window->isOpen()) {
        mutex.lock();
        window->clear(); 
        int matrixElements = videO::NUM_LAYERS * videO::NUM_NEURONS;        
        for (int i = 0; i < videO::globalNetwork->layers.size(); i++) {
            for (int j = 0; j < videO::globalNetwork->layers[i]->neurons.size(); j++) {
                double radius = videO::globalNetwork->layers[i]->neurons[j]->activation * matrixElements/3;
                if (radius > 100) {
                    radius = 100;
                }
                if (radius < 0) {
                    radius = 0;
                }
                sf::CircleShape circle(radius);

                circle.setPosition(matrixElements*i + matrixElements/2 - radius, matrixElements*j + matrixElements/2 - radius);
                if (videO::globalNetwork->layers[i]->neurons[j]->firing) {
                    sf::Color color(255, 0, 0, 255);
                    circle.setFillColor(color);
                }
                else {
                    Neuron* neuron = videO::globalNetwork->layers[i]->neurons[j];
                    sf::Color color(neuron->activation*63, neuron->activation*180+50, -neuron->activation*127+177, 255);
                    circle.setFillColor(color);
                }
                window->draw(circle);
            }
        }
        // Load a font
        sf::Font font;
        if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
            // error handling
        }

        // Create a text object
        sf::Text text;

        // Set the string to display the current pattern
        text.setString("Current pattern: " + std::to_string(videO::globalNetwork->target));

        // Set the font
        text.setFont(font);

        // Set the character size
        text.setCharacterSize(18); // in pixels, not points!

        // Set the color
        text.setFillColor(sf::Color::Red);

        // Set the text style
        text.setStyle(sf::Text::Bold | sf::Text::Underlined);

        // Set the position
        text.setPosition(0, 0);

        // Draw the text
        window->draw(text);

        //poll for events
        sf::Event event;
        while (window->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window->close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    videO::globalNetwork->setWeights(0.1);
                }
                if (event.key.code == sf::Keyboard::Enter) {
                    videO::globalNetwork->setWeights(0.01);
                }
                //keypad A
                if (event.key.code == sf::Keyboard::A) {
                    videO::globalNetwork->setWeights(0.001);
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
                if (event.key.code == sf::Keyboard::Num5) {
                    videO::globalNetwork->setTarget(5);
                }
                if (event.key.code == sf::Keyboard::Num6) {
                    videO::globalNetwork->setTarget(6);
                }
                if (event.key.code == sf::Keyboard::Num7) {
                    videO::globalNetwork->setTarget(7);
                }
                if (event.key.code == sf::Keyboard::Num8) {
                    videO::globalNetwork->setTarget(8);
                }
                if (event.key.code == sf::Keyboard::Num9) {
                    videO::globalNetwork->setTarget(9);
                }
                if (event.key.code == sf::Keyboard::Num0) {
                    videO::globalNetwork->setTarget(0);
                }
            }
        }

        window->display();
        mutex.unlock();
    }
}

int main() {

    std::cout << "Init:\n" << std::endl;
    videO::globalNetwork = new videO::Network(videO::NUM_LAYERS, videO::NUM_NEURONS);
    std::cout << "Network created" << std::endl;
    videO::globalNetwork->setWeights(0.01);
    std::cout << "Weights set" << std::endl;
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Neural Network");
    std::cout << "Window created" << std::endl;
    audiO::setupArrays();
    std::cout << "Arrays created" << std::endl;
    audiO::generateNoteMap(); // a map that will be used to map the note_matrix index to a frequency
    std::cout << "Note map generated" << std::endl;
    audiO::generateFreqs(); // a vector of frequencies that will be used to generate the sine waves
    std::cout << "Freqs generated" << std::endl;
    audiO::generateSeconds(); // a constant for now, but could be used to change the length of the note
    std::cout << "Seconds generated" << std::endl;
    videO::firingNeurons = std::vector<videO::Neuron*>();
    audiO::alsaSetup();
    std::cout << "Alsa setup" << std::endl;
    audiO::global_synth = new audiO::Synth();
    std::cout << "Synth created" << std::endl;
    std::thread display_thread(videO::display, &window);
    std::cout << "Display thread created" << std::endl;
    audiO::start_audio();
    std::cout << "Audio started" << std::endl;
    videO::nt_running = true;
    std::thread t0(videO::neuronThread);
    t0.detach();
    std::cout << "Neuron thread created" << std::endl;
    display_thread.join();
    videO::nt_running = false;
    audiO::stop_audio();
    audiO::freeArrays();
    delete videO::globalNetwork;
    delete audiO::global_synth;
    return 0;
}




