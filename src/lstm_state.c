#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm_private.h"

#include "debug.h"

void lstm_state_restore(lstm_state_t lstmState, lstm_t lstm)
{
	int i, j;
	int indexTmp;

	struct LSTM_LAYER* layerRef;

	LOG("enter");

	// Get reference
	layerRef = lstm->layerList;
	assert(layerRef != NULL);

	// Set indexTmp to last hidden layer index
	indexTmp = lstm->config.layers - 2;

	// Save hidden state
	for(i = 0; i < layerRef[indexTmp].nodeCount; i++)
	{
		layerRef[indexTmp].nodeList[i].output = lstmState->hidden[i];
	}

	// Save cell value
	for(i = 0; i < indexTmp; i++)
	{
		for(j = 0; j < layerRef[i + 1].nodeCount; j++)
		{
			layerRef[i + 1].nodeList[j].cell = lstmState->cell[i][j];
		}
	}

	LOG("exit");
}

void lstm_state_save(lstm_state_t lstmState, lstm_t lstm)
{
	int i, j;
	int indexTmp;

	struct LSTM_LAYER* layerRef;

	LOG("enter");

	// Get reference
	layerRef = lstm->layerList;
	assert(layerRef != NULL);

	// Set indexTmp to last hidden layer index
	indexTmp = lstm->config.layers - 2;

	// Save hidden state
	for(i = 0; i < layerRef[indexTmp].nodeCount; i++)
	{
		lstmState->hidden[i] = layerRef[indexTmp].nodeList[i].output;
	}

	// Save cell value
	for(i = 0; i < indexTmp; i++)
	{
		for(j = 0; j < layerRef[i + 1].nodeCount; j++)
		{
			lstmState->cell[i][j] = layerRef[i + 1].nodeList[j].cell;
		}
	}

	LOG("exit");
}

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

	lstm_alloc(tmpStatePtr->cell, hLayers, double*, ret, ERR);
	for(i = 0; i < hLayers; i++)
	{
		lstm_alloc(tmpStatePtr->cell[i], lstmCfg->nodeList[i + 1], double, ret, ERR);
	}

	lstm_alloc(tmpStatePtr->hidden, lstmCfg->nodeList[hLayers], double, ret, ERR);

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

	lstm_free(statePtr->hidden);

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
