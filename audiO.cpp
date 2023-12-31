#include "audiO.h"
#include <alsa/asoundlib.h>
#include <cmath>
#include <thread>

const int MATRIX_X_SIZE = 10;
const int MATRIX_Y_SIZE = 10;
const int MATRIX_ELEMENTS = MATRIX_X_SIZE * MATRIX_Y_SIZE;
const int NUM_SINES = 10;
const int NUM_SECONDS = 5;
const int SAMPLE_RATE = 44100;
const int NUM_CHANNELS = 2;
const int NUM_FRAMES = 32;
const int SIZE = NUM_FRAMES * NUM_CHANNELS * 2;
const int PLAY_LOOPS = 100;

char* generateSineWaves(bool** note_matrix, char* buffer, int numSines, int* freqs, int* seconds){
    char*** array = fill3DArrayWithSoundingSines(note_matrix, MATRIX_X_SIZE, MATRIX_Y_SIZE, freqs);
    int noteIndex = 0;
    for (int i; i < MATRIX_Y_SIZE; i++){
        for (int j; j < MATRIX_X_SIZE; j++){
            for (int k; k < SIZE; k++){
                buffer[k] += array[i][j][k];
            }
        }
    }
    return buffer;
}


int* generateFreqs(int size){
    int* array = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++){
        array[i] = 440*pow(2, (i - (NUM_SINES/2)) * 5.0f / 12.0f);
    }
    return array;
}

int* generateSeconds(int size){
    int* array = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++){
        array[i] = NUM_SECONDS;
    }
    return array;
}


bool** generateNoteMatrix(int xSize, int ySize){
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

char*** fill3DArrayWithSoundingSines(bool** array, int xSize, int ySize, int* freqs){
    char*** array2 = (char***)calloc(xSize, sizeof(char**));
    for (int i = 0; i < xSize; i++){
        array2[i] = (char**)calloc(ySize, sizeof(char*));
        for (int j = 0; j < ySize; j++){
            array2[i][j] = (char*)calloc(SIZE, sizeof(char));
            if (array[i][j] == true){
                for (int k = 0; k < SIZE; k++){
                    array2[i][j][k] = (char) (sin((double)k/((double)SIZE/freqs[i*j])*M_PI*2.)*127+128);
                }
            }
            else{
                for (int k = 0; k < SIZE; k++){
                    array2[i][j][k] = 0;
                }
            }
        }
    }
    return array2;
}

void play_alsa(char* buffer, int size, snd_pcm_t *handle, int rc){
    rc = snd_pcm_writei(handle, buffer, size);
    if (rc == -EPIPE) {
        // EPIPE means underrun
        fprintf(stderr, "underrun occurred\n");
        snd_pcm_prepare(handle);
    } else if (rc < 0) {
        fprintf(stderr, "error from writei: %s\n", snd_strerror(rc));
    }  else if (rc != (int)size) {
        fprintf(stderr, "short write, write %d frames\n", rc);
    }
}

void play_alsa_thread(char* buffer, int size, snd_pcm_t *handle, int rc){
    for (int i = 0; i < PLAY_LOOPS; i++){
        play_alsa(buffer, size, handle, rc);
    }
}

int main(){
    //testprogram
    bool** note_matrix = generateNoteMatrix(MATRIX_X_SIZE, MATRIX_Y_SIZE);//test matrix
    char* buffer = (char*)malloc(SIZE);
    snd_pcm_t *handle;
    int rc;
    snd_pcm_hw_params_t *params;
    uint val;
    int dir;
    snd_pcm_uframes_t frames;
    
    // open pcm device
    int err = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
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
    snd_pcm_hw_params_set_channels(handle, params, NUM_CHANNELS);
    
    // 44100 bits/second sampling rate (CD quality)
    val = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(handle, params, &val, &dir);
    
    // set period size to 32 frames
    frames = NUM_FRAMES;
    
    // write the parameters to the driver
    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0) {
        fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(rc));
        exit(1);
    }

    //sound generation
    int* freqs = generateFreqs(MATRIX_ELEMENTS);
    int* seconds = generateSeconds(MATRIX_ELEMENTS);
    while(1){
        buffer = generateSineWaves(note_matrix, buffer, NUM_SINES, freqs, seconds);
        play_alsa(buffer, SIZE, handle, rc);
    }
    return 0;
}