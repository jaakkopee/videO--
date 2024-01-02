#include <alsa/asoundlib.h>
#include <cmath>

int main(){
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
    err = snd_pcm_open(&handle, "hw:0,0", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        printf("Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(handle, params);
    snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(handle, params, 2);
    val = 44100;
    snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);
    frames = 32;
    snd_pcm_hw_params_set_period_size_near(handle, params, &frames, &dir);
    // write the parameters to the driver
    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0) {
        printf("Unable to set hw parameters: %s\n", snd_strerror(rc));
        exit(EXIT_FAILURE);
    }
    // allocate buffer to hold single period
    snd_pcm_hw_params_get_period_size(params, &frames, &dir);
    size = frames * 4; // 2 bytes/sample, 2 channels
    buffer = (char *) malloc(size);
    // fill buffer with sine wave
    for (i = 0; i < frames; i++) {
        buffer[4*i] = (char) (32767*sin((2*M_PI*freq*i)/44100)); // left
        buffer[4*i+1] = buffer[4*i]; // right

    }
    // write one period
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
    // fill buffer with sine wave
    for (i = 0; i < frames; i++) {
        buffer[4*i] = (char) (32767*sin((2*M_PI*freq2*i)/44100)); // left
        buffer[4*i+1] = buffer[4*i]; // right

    }
    // write one period
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
    return 0;
}
