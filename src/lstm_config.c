#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "lstm_private.h"
#include "lstm_builtin_math.h"

#include "debug.h"

#define LSTM_DEFAULT_INPUTS		1
#define LSTM_DEFAULT_OUTPUTS	1
#define LSTM_DEFAULT_HLAYERS	1
#define LSTM_DEFAULT_NODES		16
#define LSTM_DEFAULT_LRATE		0.01
#define LSTM_DEFAULT_MCOEF		0.1

int lstm_config_clone(lstm_config_t* lstmCfgPtr, const lstm_config_t lstmCfgSrc)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_CONFIG_STRUCT* tmpLstmCfg;

	LOG("enter");

	// Memory allocation: lstm config struct
	tmpLstmCfg = malloc(sizeof(struct LSTM_CONFIG_STRUCT));
	if(tmpLstmCfg == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto RET;
	}

	// Zero memory
	lstm_config_zeromem(tmpLstmCfg);

	// Clone config struct
	ret = lstm_config_struct_clone(tmpLstmCfg, lstmCfgSrc);
	if(ret != LSTM_NO_ERROR)
	{
		goto ERR;
	}

	// Assign value
	*lstmCfgPtr = tmpLstmCfg;

	goto RET;

ERR:
	free(tmpLstmCfg);

RET:
	LOG("exit");
	return ret;
}

int lstm_config_struct_clone(struct LSTM_CONFIG_STRUCT* dst, const struct LSTM_CONFIG_STRUCT* src)
{
	int ret = LSTM_NO_ERROR;
	int* tmpNodeList;

	LOG("enter");

	// Memory allocation: tmp node list
	tmpNodeList = calloc(src->layers, sizeof(int));
	if(tmpNodeList == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto RET;
	}

	// Copy memory
	memcpy(tmpNodeList, src->nodeList, src->layers * sizeof(int));

	// Assign values
	*dst = *src;
	dst->nodeList = tmpNodeList;

RET:
	LOG("exit");
	return ret;
}

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
	int indexTmp;
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
	indexTmp = (preLayers > 2) ? preLayers - 1 : 1;
	for(i = indexTmp; i < layers - 1; i++)
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

int lstm_config_set_hidden_nodes(lstm_config_t lstmCfg, int hiddenLayerIndex, int hiddenNodes)
{
	int ret = LSTM_NO_ERROR;
	int layerIndex;

	LOG("enter");

	// Checking
	layerIndex = hiddenLayerIndex + 1;
	if(layerIndex >= lstmCfg->layers || hiddenNodes <= 0)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		// Set value
		assert(lstmCfg->nodeList != NULL);
		lstmCfg->nodeList[layerIndex] = hiddenNodes;
	}

	LOG("exit");

	return ret;
}

int lstm_config_get_hidden_nodes(lstm_config_t lstmCfg, int hiddenLayerIndex)
{
	int ret;
	int layerIndex;

	LOG("enter");

	// Checking
	layerIndex = hiddenLayerIndex + 1;
	if(layerIndex >= lstmCfg->layers)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		// Get value
		assert(lstmCfg->nodeList != NULL);
		ret = lstmCfg->nodeList[layerIndex];
	}

	LOG("exit");

	return ret;
}

int lstm_config_set_input_transfer_func(lstm_config_t lstmCfg, int tFuncID)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	// Checking
	if(tFuncID >= LSTM_TFUNC_AMOUNT || tFuncID < 0)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		lstmCfg->inputTFunc = tFuncID;
	}

	LOG("exit");

	return ret;
}

int lstm_config_get_input_transfer_func(lstm_config_t lstmCfg)
{
	return lstmCfg->inputTFunc;
}

int lstm_config_set_output_transfer_func(lstm_config_t lstmCfg, int tFuncID)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	// Checking
	if(tFuncID >= LSTM_TFUNC_AMOUNT || tFuncID < 0)
	{
		ret = LSTM_INVALID_ARG;
	}
	else
	{
		lstmCfg->outputTFunc = tFuncID;
	}

	LOG("exit");

	return ret;
}

int lstm_config_get_output_transfer_func(lstm_config_t lstmCfg)
{
	return lstmCfg->outputTFunc;
}

void lstm_config_set_learning_rate(lstm_config_t lstmCfg, double lRate)
{
	lstmCfg->lRate = lRate;
}

double lstm_config_get_learning_rate(lstm_config_t lstmCfg)
{
	return lstmCfg->lRate;
}

void lstm_config_set_momentum_coef(lstm_config_t lstmCfg, double mCoef)
{
	lstmCfg->mCoef = mCoef;
}

double lstm_config_get_momentum_coef(lstm_config_t lstmCfg)
{
	return lstmCfg->mCoef;
}

void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfg)
{
	LOG("enter");

	memset(lstmCfg, 0, sizeof(struct LSTM_CONFIG_STRUCT));

	LOG("exit");
}

int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfg)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	// Zero memory
	lstm_config_zeromem(lstmCfg);

	// Set inputs, outputs
	lstm_config_set_inputs(lstmCfg, LSTM_DEFAULT_INPUTS);
	lstm_config_set_outputs(lstmCfg, LSTM_DEFAULT_OUTPUTS);

	// Set layers
	ret = lstm_config_set_hidden_layers(lstmCfg, LSTM_DEFAULT_HLAYERS);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Set transfer function
	lstmCfg->gateTFunc = LSTM_SIGMOID;
	lstm_config_set_input_transfer_func(lstmCfg, LSTM_HYPERBOLIC_TANGENT);
	lstm_config_set_output_transfer_func(lstmCfg, LSTM_HYPERBOLIC_TANGENT);

	// Set learning and momentum coef
	lstm_config_set_learning_rate(lstmCfg, LSTM_DEFAULT_LRATE);
	lstm_config_set_momentum_coef(lstmCfg, LSTM_DEFAULT_MCOEF);

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

	if(lstmCfg != NULL)
	{
		lstm_config_struct_delete(lstmCfg);
		free(lstmCfg);
	}

	LOG("exit");
}

void lstm_config_struct_delete(struct LSTM_CONFIG_STRUCT* lstmCfg)
{
	LOG("enter");

	// Free memory
	lstm_free(lstmCfg->nodeList);

	// Zero memory
	lstm_config_zeromem(lstmCfg);

	LOG("exit");
}

