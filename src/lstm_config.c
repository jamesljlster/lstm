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

int lstm_config_set_inputs(lstm_config_t lstmCfg, int inputs)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	// Checking
	if(inputs <= 0)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		// Set value
		lstmCfg->inputs = inputs;
		if(lstmCfg->nodeList != NULL)
		{
			lstmCfg->nodeList[0] = inputs;
		}
	}

	LOG("exit");
	return ret;
}

int lstm_config_get_inputs(lstm_config_t lstmCfg)
{
	return lstmCfg->inputs;
}

int lstm_config_set_outputs(lstm_config_t lstmCfg, int outputs)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	// Checking
	if(outputs <= 0)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		// Set value
		lstmCfg->outputs = outputs;
		if(lstmCfg->nodeList != NULL)
		{
			lstmCfg->nodeList[lstmCfg->layers - 1] = outputs;
		}
	}

	LOG("exit");
	return ret;
}

int lstm_config_get_outputs(lstm_config_t lstmCfg)
{
	return lstmCfg->outputs;
}

int lstm_config_set_hidden_layers(lstm_config_t lstmCfg, int hiddenLayers)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int preLayers, layers;

	int* tmpNodeList = NULL;

	LOG("enter");

	// Checking
	if(hiddenLayers <= 0)
	{
		ret = LSTM_INVALID_ARG;
		goto RET;
	}

	// Find layers
	preLayers = lstmCfg->layers;
	layers = hiddenLayers + 2;
	if(preLayers == layers)
	{
		// Nothing need to do
		goto RET;
	}

	// Reallocate node list
	tmpNodeList = realloc(lstmCfg->nodeList, sizeof(int) * layers);
	if(tmpNodeList == NULL)
	{
		goto RET;
	}

	// Set nodes
	for(i = preLayers - 1; i < layers - 1; i++)
	{
		tmpNodeList[i] = LSTM_DEFAULT_NODES;
	}
	tmpNodeList[0] = lstmCfg->inputs;
	tmpNodeList[layers - 1] = lstmCfg->outputs;

	// Assign values
	lstmCfg->nodeList = tmpNodeList;
	lstmCfg->layers = layers;

RET:
	LOG("exit");
	return ret;
}

int lstm_config_get_hidden_layers(lstm_config_t lstmCfg)
{
	if(lstmCfg->layers > 2)
	{
		return lstmCfg->layers - 2;
	}
	else
	{
		return 0;
	}
}

void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	LOG("enter");

	memset(lstmCfgPtr, 0, sizeof(struct LSTM_CONFIG_STRUCT));

	LOG("exit");
}

int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	int i;
	int ret = LSTM_NO_ERROR;

	LOG("enter");

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
	LOG("exit");
	return ret;
}

int lstm_config_create(lstm_config_t* lstmCfgPtr)
{
	int iResult;
	int ret = LSTM_NO_ERROR;

	struct LSTM_CONFIG_STRUCT* cfgPtr;

	LOG("enter");

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
	LOG("exit");
	return ret;
}

void lstm_config_delete(lstm_config_t lstmCfg)
{
	LOG("enter");

	lstm_config_delete_struct(lstmCfg);
	free(lstmCfg);

	LOG("exit");
}

void lstm_config_delete_struct(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	LOG("enter");

	if(lstmCfgPtr->nodeList != NULL)
	{
		free(lstmCfgPtr->nodeList);
	}

	LOG("exit");
}

