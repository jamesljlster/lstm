#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm_private.h"

#include "debug.h"

int lstm_state_create(lstm_state_t* lstmStatePtr, lstm_config_t lstmCfg)
{
	int i, hLayers;
	int ret = LSTM_NO_ERROR;
	struct LSTM_STATE_STRUCT* tmpStatePtr = NULL;

	LOG("enter");

	// Memory allocation
	lstm_alloc(tmpStatePtr, 1, struct LSTM_STATE_STRUCT, ret, ERR);

	// Clone config
	lstm_run(lstm_config_struct_clone(&tmpStatePtr->config, lstmCfg), ret, ERR);

	// Allocate vector list
	assert(lstmCfg->layers >= 3);
	hLayers = lstmCfg->layers - 2;
	lstm_alloc(tmpStatePtr->cell, hLayers, double, ret, ERR);
	lstm_alloc(tmpStatePtr->hidden, hLayers, double, ret, ERR);
	for(i = 0; i < hLayers; i++)
	{
		lstm_alloc(tmpStatePtr->cell[i], lstmCfg->nodeList[i + 1], double, ret, ERR);
		lstm_alloc(tmpStatePtr->hidden[i], lstmCfg->nodeList[i + 1], double, ret, ERR);
	}

	// Assign value
	*lstmStatePtr = tmpStatePtr;

	goto RET;

ERR:
	lstm_state_delete(tmpStatePtr);

RET:
	LOG("exit");
	return ret;
}

void lstm_state_struct_delete(struct LSTM_STATE_STRUCT* statePtr)
{
	int i, hLayers;

	LOG("enter");

	hLayers = statePtr->config.layers - 2;

	// Free memory
	if(statePtr->cell != NULL)
	{
		for(i = 0; i < hLayers; i++)
		{
			lstm_free(statePtr->cell[i]);
		}
		lstm_free(statePtr->cell);
	}

	if(statePtr->hidden != NULL)
	{
		for(i = 0; i < hLayers; i++)
		{
			lstm_free(statePtr->hidden[i]);
		}
		lstm_free(statePtr->hidden);
	}

	// Delete config
	lstm_config_struct_delete(&statePtr->config);

	// Zero memory
	memset(statePtr, 0, sizeof(struct LSTM_STATE_STRUCT));

	LOG("exit");
}

void lstm_state_delete(lstm_state_t lstmState)
{
	LOG("enter");

	if(lstmState != NULL)
	{
		lstm_state_struct_delete(lstmState);
		lstm_free(lstmState);
	}

	LOG("exit");
}
