#ifndef VIDEO_H
#define VIDEO_H

#include <alsa/asoundlib.h>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <vector>
#include <unordered_map>

namespace videO{
    const double learning_rate = 0.01;
    const double globalThreshold = 0.99999;
    const int len_at = 2^10;
    const int NUM_NEURONS = 10;
    const int NUM_LAYERS = 10;
    class Connection;
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

    void fireThread(Neuron* neuron);

    class Connection {
        public:
            Neuron* neuron_from;
            Neuron* neuron_to;
            double weight;

            Connection(Neuron* n1, Neuron* n2, double w);

            void update();
    };

    class Network;
    class Layer {
        public:
            std::vector<Neuron*> neurons;
            std::vector<Connection*> connections;
            Network* network;

            Layer(Network* net);

            void addNeuron(Neuron* neuron);

            void addConnection(Connection* connection);

            void update();

    };

    class Network{
        public:
            std::vector<Layer*> layers;
            std::vector<Connection*> connections;
            int target;

            Network(int num_layers, int num_neurons);
            
            void addLayer(Layer* layer);

            void addConnection(Connection* connection);

            void connect();

            void setTarget(int target);

            void update();

            void backpropWithTarget();

            void setWeights(double weight);

            void connectSelf(Layer* layer);

            void connectLayers();

            void connectWithLayer(Layer* layer1, Layer* layer2);

            Network* getNetwork();

    };

    double sigmoid(double x);
    void neuronThread();
    bool nt_running = false;
    void display(sf::RenderWindow* window);
    Network* globalNetwork;


}
namespace audiO{
    const int MATRIX_X_SIZE = 10;
    const int MATRIX_Y_SIZE = 10;
    const int MATRIX_ELEMENTS = MATRIX_X_SIZE * MATRIX_Y_SIZE;
    const int NUM_SINES = 10;
    const float NUM_SECONDS = 0.5;
    const float SAMPLE_RATE = 44100;
    const int NUM_CHANNELS = 1;
    const int NUM_FRAMES = (int)(((float)(SAMPLE_RATE) * NUM_SECONDS));
    const int SIZE = NUM_FRAMES * NUM_CHANNELS * 2;
    const int PLAY_LOOPS = 100;
    const int SAMPLE_MAX = 32767;
    const int SAMPLE_MIN = -32768;
    const float AMPLITUDE_NEURON = ((float)SAMPLE_MAX / (float)MATRIX_ELEMENTS);

    snd_pcm_t *handle_alsa;
    int rc_alsa;
    snd_pcm_uframes_t frames;
    std::vector<float> audio_float_buffer;
    short* audiobuffer;
    bool running = false;
    //float*** sinewaves;
    std::vector<float> freqs;
    std::vector<float> seconds;
    std::vector<std::vector<bool>> note_matrix;
    std::unordered_map<int, float> note_map; // maps note_matrix index to frequency

    void generateNoteMap();

    std::vector<std::vector<bool>> generateNoteMatrix();

    std::vector<std::vector<bool>> fireToBool();

    void setOscAmpsWithNeuronActivations();

    //short* generateSineWaves(bool** note_matrix, short* buffer, int numSines, int* freqs, int* seconds);

    std::vector<float> generateSineWaves(); // sound synthesis

    float sigmoidSaturator(float x);

    class Oscillator {
        public:
            float freq;
            float amp;
            float phase;
            float phase_increment;
            float* buffer;
            int buffer_size;
            int buffer_index;
            bool running;

            Oscillator(float freq, float amp, float phase);

            ~Oscillator();

            float getSample();

            void setFrequency(float frequency);

            void setAmplitude(float amplitude);

            void setPhase(float phase);

            void setPhaseIncrement(float phase_increment);

            void setBuffer(float* buffer);

            void setBufferSize(int buffer_size);

            void setBufferIndex(int buffer_index);

            void setRunning(bool running);

            float getFrequency();

            float getAmplitude();

            float getPhase();

            float getPhaseIncrement();

            float* getBuffer();

            int getBufferSize();

            int getBufferIndex();

            bool getRunning();

    };

    class OscillatorBank {
        public:
            int num_oscillators;
            Oscillator** oscillators;
            float* buffer;
            int buffer_size;
            int buffer_index;
            bool running;

            OscillatorBank();

            ~OscillatorBank();

            float getSample();

            void setNumOscillators(int num_oscillators);

            void setOscillators(Oscillator* oscillators);

            void setBuffer(float* buffer);

            void setBufferSize(int buffer_size);

            void setBufferIndex(int buffer_index);

            void setRunning(bool running);

            int getNumOscillators();

            Oscillator* getOscillators();

            float* getBuffer();

            int getBufferSize();

            int getBufferIndex();

            bool getRunning();

            void generateWave();

            void generateSineWave();

            void generateSawWave();

            void generateSquareWave();

            void generateTriangleWave();

            void generateNoise();

            void generatePulseWave();

    };

    OscillatorBank* global_oscillator_bank;

    void play_alsa();

    void play_alsa_thread();

    void audio_thread();

    void rampAmplitudeThread(Oscillator* osc, videO::Neuron* neuron);

    void setupArrays();

    void freeArrays();

    std::vector<float> generateFreqs();

    std::vector<float> generateSeconds();

    void alsaSetup();

    void start_audio();

    void stop_audio();

}




#endif