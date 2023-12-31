#include "Tonic/Tonic.h"
#include <iostream>
#include <alsa/asoundlib.h>
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

// alsa setup
int err;
unsigned int i;
snd_pcm_t *handle;
char *buffer;
int size;
int dir;
snd_pcm_uframes_t frames;
snd_pcm_hw_params_t *params;
int rc;
int size2;
uint val;
int freq = 440;
int seconds = 5;
int freq2 = 550;
int seconds2 = 5;
int freq3 = 660;
int seconds3 = 5;
int freq4 = 770;
int seconds4 = 5;
int freq5 = 880;
int seconds5 = 5;
int freq6 = 990;
int seconds6 = 5;

// open pcm device
err = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
if (err < 0) {
    printf("Playback open error: %s\n", snd_strerror(err));
    exit(EXIT_FAILURE);
}

// allocate a hardware parameters object
snd_pcm_hw_params_alloca(&params);

// fill it in with default values
snd_pcm_hw_params_any(handle, params);

// set the desired hardware parameters
// INTERLEAVED is the only mode supported by most sound cards
snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);

// set the sample format
snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);

// set the count of channels
snd_pcm_hw_params_set_channels(handle, params, 2);

// set the sample rate
val = 44100;
snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);

// set the buffer time
snd_pcm_hw_params_set_buffer_time_near(handle, params, &val, &dir);

// write the parameters to device
err = snd_pcm_hw_params(handle, params);

// prepare interface for use
snd_pcm_prepare(handle);

// allocate buffer to hold single period
snd_pcm_hw_params_get_period_size(params, &frames, &dir);

// we want to loop for 5 seconds
snd_pcm_hw_params_get_period_time(params, &val, &dir);

// 5 seconds in microseconds divided by period time
size = val * 2 * 2; // 2 bytes/sample, 2 channels

buffer = (char *) malloc(size);

// loop for 5 seconds
snd_pcm_hw_params_get_period_time(params, &val, &dir);


// Tonic setup
Tonic::setSampleRate(44100);
Tonic::TonicFrames data(1024, 2);

while (1) {
    Tonic::Generator gen = Tonic::SineWave().freq(440);
    Tonic::Generator gen2 = Tonic::SineWave().freq(550);
    Tonic::Generator gen3 = Tonic::SineWave().freq(660);
    Tonic::Generator gen4 = Tonic::SineWave().freq(770);
    Tonic::Generator gen5 = Tonic::SineWave().freq(880);
    Tonic::Generator gen6 = Tonic::SineWave().freq(990);
    Tonic::Generator gen7 = Tonic::SineWave().freq(1100);
    Tonic::Generator gen8 = Tonic::SineWave().freq(1210);
    Tonic::Generator gen9 = Tonic::SineWave().freq(1320);
    Tonic::Generator gen10 = Tonic::SineWave().freq(1430);

    for (unsigned int i=0; i<data.size(); i++){
        gen.tick(data, );
        gen2.tick(data);
        gen3.tick(data);
        gen4.tick(data);
        gen5.tick(data);
        gen6.tick(data);
        gen7.tick(data);
        gen8.tick(data);
        gen9.tick(data);
        gen10.tick(data);
    }

    // write the data to the device
    rc = snd_pcm_writei(handle, buffer, frames);
    if (rc == -EPIPE) {
        // EPIPE means underrun
        fprintf(stderr, "underrun occurred\n");
        snd_pcm_prepare(handle);
    } else if (rc < 0) {
        fprintf(stderr, "error from writei: %s\n", snd_strerror(rc));
    }  else if (rc != (int)frames) {
        fprintf(stderr, "short write, write %d frames\n", rc);
    }


}
