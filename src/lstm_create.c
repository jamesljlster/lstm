#include <stdlib.h>

#include "lstm.h"
#include "lstm_private.h"

int lstm_config_create(lstm_config_t* lstmCfgPtr)
{
	int iResult;
	int ret = LSTM_NO_ERROR;

	struct LSTM_CONFIG_STRUCT* cfgPtr;

	// Memory allocation
	cfgPtr = malloc(sizeof(struct LSTM_CONFIG_STRUCT));
	if(cfgPtr == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto RET;
	}

	// Initial lstm config
	iResult = lstm_config_init(cfgPtr);
	if(iResult != LSTM_NO_ERROR)
	{
		ret = iResult;
		goto ERR;
	}

	// Assign value
	*lstmCfgPtr = cfgPtr;

	goto RET;

ERR:
	free(cfgPtr);

RET:
	return ret;
}
