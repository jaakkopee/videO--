#include "audiO.h"

using namespace std;


const double NOTE_DURATION = 0.5;
const int NOTE_DURATION_SAMPLES = (int)(NOTE_DURATION * SAMPLE_RATE);
const double NOTE_VOLUME = 0.25;
const int N_OSCILLATORS = 100;

double getFrequency(int note) {
    return 440.0 * pow(2.0, (note - 69.0) / 12.0);
}

// Oscillator class
Oscillator::Oscillator(double frequency, double amplitude, double phase) {
    this->frequency = frequency;
    this->amplitude = amplitude;
    this->phase = phase;
    this->sample = 0.0;
    this->sample_index = 0;
}

void Oscillator::setFrequency(double frequency) {
    this->frequency = frequency;
}

void Oscillator::setAmplitude(double amplitude) {
    this->amplitude = amplitude;
}

void Oscillator::setPhase(double phase) {
    this->phase = phase;
}

double Oscillator::getFrequency() {
    return this->frequency;
}

double Oscillator::getAmplitude() {
    return this->amplitude;
}

double Oscillator::getPhase() {
    return this->phase;
}

double Oscillator::getSample() {
    this->sample = this->amplitude * sin(2 * M_PI * this->frequency * this->sample_index / SAMPLE_RATE + this->phase);
    this->sample_index++;
    this->sample_index %= (int)SAMPLE_RATE;
    return this->sample;
}

EnvelopeGenerator::EnvelopeGenerator(double attack, double decay, double sustain, double release) {
    this->attack = attack;
    this->decay = decay;
    this->sustain = sustain;
    this->release = release;
}

void EnvelopeGenerator::setAttack(double attack) {
    this->attack = attack;
}

void EnvelopeGenerator::setDecay(double decay) {
    this->decay = decay;
}

void EnvelopeGenerator::setSustain(double sustain) {
    this->sustain = sustain;
}

void EnvelopeGenerator::setRelease(double release) {
    this->release = release;
}

double EnvelopeGenerator::getAttack() {
    return this->attack;
}

double EnvelopeGenerator::getDecay() {
    return this->decay;
}

double EnvelopeGenerator::getSustain() {
    return this->sustain;
}

double EnvelopeGenerator::getRelease() {
    return this->release;
}

vector<double> EnvelopeGenerator::getBuffer() {
    vector<double> buffer;
    for (int i = 0; i < NOTE_DURATION_SAMPLES; i++) {
        double sample = 0.0;
        if (i < this->attack * SAMPLE_RATE) {
            sample = i / (this->attack * SAMPLE_RATE);
        } else if (i < (this->attack + this->decay) * SAMPLE_RATE) {
            sample = 1.0 - (1.0 - this->sustain) * (i - this->attack * SAMPLE_RATE) / (this->decay * SAMPLE_RATE);
        } else if (i < (this->attack + this->decay + NOTE_DURATION) * SAMPLE_RATE) {
            sample = this->sustain;
        } else if (i < (this->attack + this->decay + NOTE_DURATION + this->release) * SAMPLE_RATE) {
            sample = this->sustain * (1.0 - (i - (this->attack + this->decay + NOTE_DURATION) * SAMPLE_RATE) / (this->release * SAMPLE_RATE));
        }
        buffer.push_back(sample);
    }
    return buffer;
}

Mixer::Mixer() {
    this->amplitude = 1.0;
    this->sample = 0.0;
}

void Mixer::addOscillator(Oscillator *oscillator) {
    this->oscillators.push_back(oscillator);
}

void Mixer::removeOscillator(Oscillator *oscillator) {
    for (int i = 0; i < this->oscillators.size(); i++) {
        if (this->oscillators[i] == oscillator) {
            this->oscillators.erase(this->oscillators.begin() + i);
            break;
        }
    }
}

void Mixer::setAmplitude(double amplitude) {
    this->amplitude = amplitude;
}

double Mixer::getAmplitude() {
    return this->amplitude;
}

double Mixer::getSample() {
    this->sample = 0.0;
    for (int i = 0; i < this->oscillators.size(); i++) {
        this->sample += this->oscillators[i]->getSample();
    }
    this->sample *= this->amplitude;
    return this->sample;
}

Synthesizer::Synthesizer() {
    this->mixer = new Mixer();
    for (int i = 0; i < N_OSCILLATORS; i++) {
        this->addOscillator(new Oscillator(0.0, 0.0, 0.0));
    }
    this->eventsPlaying = vector<bool>(N_OSCILLATORS, false);
    this->envelopeGenerators = vector<EnvelopeGenerator*>();
    this->sample_index = 0;
    this->samples = vector<double>();
    this->playing = false;
}


void Synthesizer::addOscillator(Oscillator *oscillator) {
    this->mixer->addOscillator(oscillator);
}

void Synthesizer::removeOscillator(Oscillator *oscillator) {
    this->mixer->removeOscillator(oscillator);
}

void Synthesizer::addEnvelopeGenerator(EnvelopeGenerator *envelopeGenerator) {
    this->envelopeGenerators.push_back(envelopeGenerator);
}

void Synthesizer::removeEnvelopeGenerator(EnvelopeGenerator *envelopeGenerator) {
    for (int i = 0; i < this->envelopeGenerators.size(); i++) {
        if (this->envelopeGenerators[i] == envelopeGenerator) {
            this->envelopeGenerators.erase(this->envelopeGenerators.begin() + i);
            break;
        }
    }
}

void Synthesizer::playEvent(int eventID) {
    vector<double> buffer = this->envelopeGenerators[eventID]->getBuffer();
    this->mutexes[eventID]->lock();
    for (int i = 0; i < buffer.size(); i++) {
        this->samples.push_back(buffer[i] * this->mixer->getSample());
    }

}

void Synthesizer::play() {
    this->playing = true;
    this->sample_index = 0;
    for (int i = 0; i < this->eventsPlaying.size(); i++) {
        this->eventsPlaying[i] = false;
    }
    for (int i = 0; i < this->envelopeGenerators.size(); i++) {
        this->mutexes.push_back(new mutex());
        this->threads.push_back(new thread(&Synthesizer::playEvent, this, i));
    }
}

void Synthesizer::stop(int eventID) {
    this->eventsPlaying[eventID] = false;
}   
    
    
void audioCallback(void *outputBuffer, unsigned long framesPerBuffer, double streamTime, RtAudioStreamStatus status, void *userData) {
    
int main() {
    Synthesizer *synthesizer = new Synthesizer();
    for (int i = 0; i < N_OSCILLATORS; i++) {
        synthesizer->addOscillator(new Oscillator(i*10, 0.01, 0.0));
    }
    for (int i = 0; i < N_OSCILLATORS; i++) {
        synthesizer->addEnvelopeGenerator(new EnvelopeGenerator(0.01, 0.1, 0.5, 0.1));
    }
    for (int i = 0; i < N_OSCILLATORS; i++) {
        synthesizer->play();
        this_thread::sleep_for(chrono::milliseconds(500));
        synthesizer->stop(i);
    }
    return 0;
}






