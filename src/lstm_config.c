#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

#define LSTM_DEFAULT_INPUTS		1
#define LSTM_DEFAULT_OUTPUTS	1
#define LSTM_DEFAULT_LAYERS		3
#define LSTM_DEFAULT_NODES		16
#define LSTM_DEFAULT_LRATE		0.01
#define LSTM_DEFAULT_MCOEF		0.1

void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	memset(lstmCfgPtr, 0, sizeof(struct LSTM_CONFIG_STRUCT));
}

int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	int i;
	int ret = LSTM_NO_ERROR;

	// Zero memory
	lstm_config_zeromem(lstmCfgPtr);

	// Allocate node list
	lstmCfgPtr->nodeList = calloc(LSTM_DEFAULT_LAYERS, sizeof(int));
	if(lstmCfgPtr->nodeList == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto RET;
	}

	// Set nodes
	for(i = 1; i < LSTM_DEFAULT_LAYERS - 1; i++)
	{
		lstmCfgPtr->nodeList[i] = LSTM_DEFAULT_NODES;
	}

	// Set inputs, outputs
	lstmCfgPtr->inputs = LSTM_DEFAULT_INPUTS;
	lstmCfgPtr->outputs = LSTM_DEFAULT_OUTPUTS;

	lstmCfgPtr->nodeList[0] = LSTM_DEFAULT_INPUTS;
	lstmCfgPtr->nodeList[LSTM_DEFAULT_LAYERS - 1] = LSTM_DEFAULT_OUTPUTS;

	// Set layers
	lstmCfgPtr->layers = LSTM_DEFAULT_LAYERS;
	
	// Set transfer function
	lstmCfgPtr->gateTFunc = LSTM_SIGMOID;
	lstmCfgPtr->inputTFunc = LSTM_HYPERBOLIC_TANGENT;
	lstmCfgPtr->outputTFunc = LSTM_HYPERBOLIC_TANGENT;

	// Set learning rate and momentum coef
	lstmCfgPtr->lRate = LSTM_DEFAULT_LRATE;
	lstmCfgPtr->mCoef = LSTM_DEFAULT_MCOEF;

RET:
	return ret;
}

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

void lstm_config_delete(lstm_config_t lstmCfg)
{
	lstm_config_delete_struct(lstmCfg);
	free(lstmCfg);
}

void lstm_config_delete_struct(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	if(lstmCfgPtr->nodeList != NULL)
	{
		free(lstmCfgPtr->nodeList);
	}
}

