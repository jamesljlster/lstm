#include <stdlib.h>

#include "lstm.h"
#include "lstm_private.h"

#define LSTM_DEFAULT_INPUTS		1
#define LSTM_DEFAULT_OUTPUTS	1
#define LSTM_DEFAULT_LAYERS		3
#define LSTM_DEFAULT_NODES		16

int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	int ret = LSTM_NO_ERROR;

	// Zero memory
	lstm_config_zeromem(lstmCfgPtr);

	// Set default value

RET:
	return ret;
}

