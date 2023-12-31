#ifndef VIDEO_H
#define VIDEO_H

#include <alsa/asoundlib.h>
namespace videO{
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
}


#endif