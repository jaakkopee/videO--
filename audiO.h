/*
This file contains definitions for classes and functions for a synthesizer that can play multiple notes at once.
The synthesizer is used by the main program to play a note when a neuron fires.
Classes needed
- oscillator with adjustable phase, frequency, and amplitude
- envelope generator with adjustable attack, decay, sustain, and release
- mixer to combine multiple oscillators
- synthesizer to serve as an interface to the mixer, oscillators, and envelope generators.
  The synth also takes care of the audio output.

Functions needed
- play a note
- stop a note
- change the frequency of a note
- change the amplitude of a note
- change the attack, decay, sustain, and release of a note
*/

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <thread>
#include <mutex>

#include "rtaudio/RtAudio.h"

using namespace std;

// Constants
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;
const double SAMPLE_RATE = 44100.0;
const double SAMPLE_PERIOD = 1.0 / SAMPLE_RATE;
const double MAX_AMP = 0.25;

//function for getting equaltempered frequency
double getFrequency(int note);

// Function prototypes
void audioCallback(void *outputBuffer, unsigned int nBufferFrames, double streamTime, RtAudioStreamStatus status, Synthesizer *synth);

// Classes
class Oscillator {
public:
    Oscillator(double frequency, double amplitude, double phase);
    void setFrequency(double frequency);
    void setAmplitude(double amplitude);
    void setPhase(double phase);
    double getFrequency();
    double getAmplitude();
    double getPhase();
    double getSample();

private:
    double frequency;
    double amplitude;
    double phase;
    double sample;
    int sample_index;
};

class EnvelopeGenerator {
public:
    EnvelopeGenerator(double attack, double decay, double sustain, double release);
    void setAttack(double attack);
    void setDecay(double decay);
    void setSustain(double sustain);
    void setRelease(double release);
    double getAttack();
    double getDecay();
    double getSustain();
    double getRelease();
    vector<double> getBuffer();

private:
    double attack;
    double decay;
    double sustain;
    double release;
};

class Mixer {
public:
    Mixer();
    void addOscillator(Oscillator *oscillator);
    void removeOscillator(Oscillator *oscillator);
    void setAmplitude(double amplitude);
    double getAmplitude();
    double getSample();

private:
    vector<Oscillator*> oscillators;
    double amplitude;
    double sample;
};

class Synthesizer {
public:
    Synthesizer();
    void addOscillator(Oscillator *oscillator);
    void removeOscillator(Oscillator *oscillator);
    void addEnvelopeGenerator(EnvelopeGenerator *envelopeGenerator);
    void removeEnvelopeGenerator(EnvelopeGenerator *envelopeGenerator);
    void playEvent(int eventID);
    void play();
    void stop(int eventID);

private:
    Mixer *mixer;
    vector<EnvelopeGenerator*> envelopeGenerators;
    vector<Oscillator*> oscillators;
    vector<thread*> threads;
    vector<mutex*> mutexes;
    vector<bool> eventsPlaying;
    double frequency;
    double amplitude;
    double attack;
    double decay;
    double sustain;
    double release;
    bool playing;
    int sample_index;
    vector<double> samples;
};




