#include <alsa/asoundlib.h>
#include <cmath>

bool** generateNoteMatrix(int xSize, int ySize);

char*** fill3DArrayWithSoundingSines(bool** array, int xSize, int ySize, int* freqs);

char* generateSineWaves(bool** note_matrix, char** array, int numSines, int* freqs, int* seconds, int numSeconds, int sampleRate, int numChannels, int numFrames, snd_pcm_t *handle, char *buffer, int size, int dir, snd_pcm_uframes_t frames, snd_pcm_hw_params_t *params, int rc, int size2, uint val);

void play_alsa(char* buffer, int size, snd_pcm_t *handle, int rc);

int* generateFreqs(int size);

int* generateSeconds(int size);