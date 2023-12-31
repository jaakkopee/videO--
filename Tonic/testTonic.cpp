#include "Tonic.h"
#include <iostream>
#include <asoundlib.h>
using namespace Tonic;

int main(int argc, char *argv[])
{
    //Tonic is a collection of signal generators and processors
TriangleWave tone1 = TriangleWave();
SineWave tone2 = SineWave();
SineWave vibrato = SineWave().freq(10);
SineWave tremolo = SineWave().freq(1);

//that you can combine using intuitive operators
Generator combinedSignal = (tone1 + tone2) * tremolo;
        
//and plug in to one another
float baseFreq = 200;
tone1.freq(baseFreq + vibrato * 10);
tone2.freq(baseFreq * 2 + vibrato * 20);

//you can also use the << operator to connect signals to a global audio output
Tonic::setSampleRate(44100);
Tonic::Synth synth;
synth.setOutputGen(combinedSignal);
while (1) {
    float buffer[kSynthesisBlockSize * 2];
    synth.fillBufferOfFloats(buffer, kSynthesisBlockSize, 2);
    write(1, buffer, kSynthesisBlockSize * 2 * sizeof(float));
}

    return 0;
}
