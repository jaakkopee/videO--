#ifndef AUDIO_H
#define AUDIO_H

#include <alsa/asoundlib.h>
#include <cmath>
namespace audiO{
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
    snd_pcm_t *handle;
    int rc;
}

bool** generateNoteMatrix(int xSize, int ySize);

char*** fill3DArrayWithSoundingSines(bool** array, int xSize, int ySize, int* freqs);

char* generateSineWaves(bool** note_matrix, char* buffer, int numSines, int* freqs, int* seconds);

void play_alsa(char* buffer, int size, snd_pcm_t *handle, int rc);

void play_alsa_thread(char* buffer);

int* generateFreqs(int size);

int* generateSeconds(int size);

void alsaSetup();

#endif