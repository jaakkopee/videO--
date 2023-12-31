// Miscellaneous parsing and error functions for use with STK demo program.
//
// Gary P. Scavone, 1999.

#include "stk/Instrmnt.h"
#include "stk/FileWvOut.h"
#include "stk/Messager.h"

int voiceByNumber(int number, stk::Instrmnt **instrument);

int voiceByName(char *name, stk::Instrmnt **instrument);

void usage(char *function);

int checkArgs(int numArgs, char *args[]);

int countVoices(int nArgs, char *args[]);

bool parseArgs(int numArgs, char *args[], stk::WvOut **output, stk::Messager& messager);
