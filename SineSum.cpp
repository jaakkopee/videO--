#include <alsa/asoundlib.h>
#include <cmath>

int main(int argc, char *argv[])
{

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
    // INTERLEAVED is the default
    snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    
    // signed 16 bit little endian format
    snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
    
    // two channels (stereo)
    snd_pcm_hw_params_set_channels(handle, params, 2);
    
    // 44100 bits/second sampling rate (CD quality)
    val = 44100;
    snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);
    
    // set period size to 32 frames
    frames = 32;
    snd_pcm_hw_params_set_period_size_near(handle, params, &frames, &dir);
    
    // write the parameters to the driver
    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0) {
        printf("Unable to set hw parameters: %s\n", snd_strerror(rc));
        exit(EXIT_FAILURE);
    }
    
    // use a buffer large enough to hold one period
    snd_pcm_hw_params_get_period_size(params, &frames, &dir);

    // 2 bytes/sample, 2 channels
    size = frames * 4;
    buffer = (char *) malloc(size);

    // we want to loop for 5 seconds
    while(1){
        // fill the buffer with sine waves
        for (i = 0; i < frames; i++) {
            buffer[4*i] = (char) (sin((double)i/((double)frames/freq)*M_PI*2.)*127+128);
            buffer[4*i+1] = (char) (sin((double)i/((double)frames/freq2)*M_PI*2.)*127+128);
            buffer[4*i+2] = (char) (sin((double)i/((double)frames/freq3)*M_PI*2.)*127+128);
            buffer[4*i+3] = (char) (sin((double)i/((double)frames/freq4)*M_PI*2.)*127+128);
            buffer[4*i+4] = (char) (sin((double)i/((double)frames/freq5)*M_PI*2.)*127+128);
            buffer[4*i+5] = (char) (sin((double)i/((double)frames/freq6)*M_PI*2.)*127+128);
        }
        
        // write the buffer to the device
        rc = snd_pcm_writei(handle, buffer, frames);
        if (rc == -EPIPE) {
            // EPIPE means underrun
            printf("Underrun occurred\n");
            snd_pcm_prepare(handle);
        } else if (rc < 0) {
            printf("Error from writei: %s\n", snd_strerror(rc));
        }  else if (rc != (int)frames) {
            printf("Short write, write %d frames\n", rc);
        }
    }

    snd_pcm_drain(handle);
    snd_pcm_close(handle);
    free(buffer);

    return 0;
}

