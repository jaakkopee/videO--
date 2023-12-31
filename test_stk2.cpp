
#define __STK_REALTIME__

#include "stk/SKINImsg.h"
#include "stk/WvOut.h"
#include "stk/Instrmnt.h"
#include "stk/JCRev.h"
#include "stk/Voicer.h"
#include "stk/Messager.h"
#include "stk/Skini.h"
#include "rtaudio/RtAudio.h"
#include "stk/Mutex.h"
#include "stk/Saxofony.h"
#include "stk/RtWvOut.h"



#include <signal.h>
#include <iostream>
#include <algorithm>
#include <cmath>
using std::min;

bool done;
static void finish(int ignore){ done = true; }
static int DELTA_CONTROL_TICKS = 64; // default sample frames between control input checks

using namespace stk;

// The TickData structure holds all the class instances and data that
// are shared by the various processing functions.
struct TickData {
  RtWvOut **wvout;
  Instrmnt **instrument;
  Voicer *voicer;
  JCRev reverb;
  Messager messager;
  Skini::Message message;
  StkFloat volume;
  StkFloat t60;
  unsigned int nWvOuts;
  int nVoices;
  int currentVoice;
  int channels;
  int counter;
  bool realtime;
  bool settling;
  bool haveMessage;
  int frequency;

  // Default constructor.
  TickData()
    : wvout(0), instrument(0), voicer(0), volume(1.0), t60(0.75),
      nWvOuts(0), nVoices(1), currentVoice(0), channels(2), counter(0),
      realtime( true ), settling( false ), haveMessage( false ) {}
};

// The Player class handles RtAudio callbacks to output sound.
class Player {
public:
  Player( TickData *data ) : tickData_( data ) {}
  ~Player() {}

  // The RtAudio Callback function.
  int tick( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
           double streamTime, RtAudioStreamStatus status, void *dataPointer);

  // This function handles control change and message parsing from the
  // GUI interface.
  void messageHandler( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                       double streamTime, RtAudioStreamStatus status );
  TickData *tickData_;

};

void Player :: messageHandler( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                               double streamTime, RtAudioStreamStatus status )
{
  register StkFloat *samples = (StkFloat *) outputBuffer;
  register unsigned int i, j;
  StkFloat sample;
  int counter, nTicks = (int) nBufferFrames;

  // Check for and load any command messages.
  stk::Skini::Message msg = tickData_->message;
  if ( tickData_->haveMessage ) {
    tickData_->messager.popMessage( msg );
    tickData_->haveMessage = false;
  }

  while ( nTicks > 0 && !done ) {

    if ( !tickData_->haveMessage ) {
      tickData_->messager.popMessage( msg );
      if ( msg.type > 0 ) {
        tickData_->counter = (long) (msg.time * Stk::sampleRate());
        tickData_->haveMessage = true;
      }
      else
        tickData_->counter = DELTA_CONTROL_TICKS;
    }

    counter = min( nTicks, tickData_->counter );
    tickData_->counter -= counter;
    for ( i=0; i<(unsigned int)counter; i++ ) {
      sample = tickData_->volume * tickData_->reverb.tick( tickData_->voicer->tick() );
      for ( j=0; j<tickData_->nWvOuts; j++ ) tickData_->wvout[j]->tick(sample);
      if ( tickData_->realtime )
        for ( int k=0; k<tickData_->channels; k++ ) *samples++ = sample;
      nTicks--;
    }
    if ( nTicks == 0 ) break;

    // Process control messages.
    if ( tickData_->haveMessage ) messageHandler( outputBuffer, inputBuffer, nBufferFrames, streamTime, status );
  }
}

TickData *globalTickData_ = new TickData;
Player *globalPlayer = new Player( globalTickData_ );
int player_tick( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                    double streamTime, RtAudioStreamStatus status, void *dataPointer ){
  register StkFloat *samples = (StkFloat *) outputBuffer;
  register unsigned int i, j;
  StkFloat sample;
  int counter, nTicks = (int) nBufferFrames;

  // Check for and load any command messages.
  stk::Skini::Message msg = globalPlayer->tickData_->message;
  if ( globalPlayer->tickData_->haveMessage ) {
    globalPlayer->tickData_->messager.popMessage( msg );
    globalPlayer->tickData_->haveMessage = false;
  }

  while ( nTicks > 0 && !done ) {

    if ( !globalPlayer->tickData_->haveMessage ) {
      globalPlayer->tickData_->messager.popMessage( msg );
      if ( msg.type > 0 ) {
        globalPlayer->tickData_->counter = (long) (msg.time * Stk::sampleRate());
        globalPlayer->tickData_->haveMessage = true;
      }
      else
        globalPlayer->tickData_->counter = DELTA_CONTROL_TICKS;
    }

    counter = min( nTicks, globalPlayer->tickData_->counter );
    globalPlayer->tickData_->counter -= counter;
    for ( i=0; i<(unsigned int)counter; i++ ) {
      sample = globalPlayer->tickData_->volume * globalPlayer->tickData_->reverb.tick( globalPlayer->tickData_->voicer->tick() );
      for ( j=0; j<globalPlayer->tickData_->nWvOuts; j++ ) globalPlayer->tickData_->wvout[j]->tick(sample);
      if ( globalPlayer->tickData_->realtime )
        for ( int k=0; k<globalPlayer->tickData_->channels; k++ ) *samples++ = sample;
      nTicks--;
    }
    if ( nTicks == 0 ) break;

    // Process control messages.
    if ( globalPlayer->tickData_->haveMessage ) globalPlayer->messageHandler( outputBuffer, inputBuffer, nBufferFrames, streamTime, status );
  }

  return 0;
}





int main(){
  // Set the global sample rate before creating class instances.
  Stk::setSampleRate( 44100.0 );
  // Set the global rawwave path before creating class instances.
  Stk::setRawwavePath( "/home/jaakko/Koodit/stk-5.0.1/rawwaves" );

  // Initialize our WvOut objects.
  globalTickData_->nWvOuts = 1;
  globalTickData_->wvout = (RtWvOut **) calloc( globalTickData_->nWvOuts, sizeof(RtWvOut *) );
  //real time output
  globalTickData_->wvout[0] = new RtWvOut( globalTickData_->channels );

  // Initialize our WvIn object.
  // globalTickData_->wvin = new FileWvIn( "impuls20.wav" );

  // Initialize our reverb.
  globalTickData_->reverb.setT60( 0.75 );

  // Initialize our Voicer.
  globalTickData_->voicer = new Voicer();
  globalTickData_->voicer->setSampleRate( Stk::sampleRate() );

  // Initialize our instruments.
  globalTickData_->nVoices = 1;
  globalTickData_->instrument = (Instrmnt **) calloc( globalTickData_->nVoices, sizeof(Instrmnt *) );
  globalTickData_->instrument[0] = new Saxofony(10.0);
  globalTickData_->instrument[0]->setFrequency( 220.0 );

  // Set the global tick data pointer for the Player.
  globalPlayer->tickData_ = globalTickData_;

  // Setup RtAudio
  RtAudio dac;
  RtAudio::StreamParameters parameters;
  parameters.deviceId = dac.getDefaultOutputDevice();
  parameters.nChannels = globalTickData_->channels;
  RtAudioFormat format = ( sizeof(StkFloat) == 8 ) ? RTAUDIO_FLOAT64 : RTAUDIO_FLOAT32;
  unsigned int bufferFrames = RT_BUFFER_SIZE;
  dac.openStream( &parameters, NULL, format, (unsigned int)Stk::sampleRate(), &bufferFrames, &player_tick, (void *)&globalPlayer );

  // Install an interrupt handler function.
  (void) signal(SIGINT, finish);

  // Resize the StkFrames object appropriately.
  StkFrames frames( (unsigned int) ( 0.5 * Stk::sampleRate() ), globalTickData_->channels );

  while( !done ) {
    // Block waiting until callback signals done.
    sleep( 1 );
  }

  // Shut down the output stream.
  dac.closeStream();

  // Cleanup.
  delete globalTickData_->voicer;
  for ( int i=0; i<globalTickData_->nVoices; i++ ) delete globalTickData_->instrument[i];
  delete globalTickData_->wvout[0];
  free( globalTickData_->wvout );
  free( globalTickData_->instrument );

  return 0;
}
