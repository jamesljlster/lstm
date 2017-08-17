#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm_private.h"
#include "lstm_builtin_math.h"

#include "debug.h"

int lstm_alloc_network(struct LSTM_STRUCT* lstm)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int nodeType, reNetSize;

	int layers;
	int* nodeList;

	struct LSTM_LAYER* layerRef = NULL;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	LOG("enter");

	// Set reference
	cfgRef = &lstm->config;
	layers = cfgRef->layers;
	nodeList = cfgRef->nodeList;

	// Checking
	if(layers < 3 || nodeList == NULL)
	{
		ret = LSTM_INVALID_ARG;
		goto RET;
	}

	// Allocate layer list
	lstm_alloc(layerRef, layers, struct LSTM_LAYER, ret, RET);
	for(i = 0; i < cfgRef->layers; i++)
	{
		// Set node type and recurrent network size
		if(i == 0)
		{
			nodeType = LSTM_INPUT_NODE;
			reNetSize = 0;
		}
		else if(i == 1)
		{
			nodeType = LSTM_FULL_NODE;
			reNetSize = nodeList[layers - 1];
		}
		else if(i == layers - 1)
		{
			nodeType = LSTM_OUTPUT_NODE;
			reNetSize = 0;
		}
		else
		{
			nodeType = LSTM_FULL_NODE;
			reNetSize = 0;
		}

		// Allocate layer struct
		ret = lstm_layer_alloc(&layerRef[i], nodeList[i], nodeType, nodeList[i], reNetSize);
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Set node count
		layerRef[i].nodeCount = nodeList[i];

		// Set activation function
		if(cfgRef->inputTFunc < 0 || cfgRef->inputTFunc >= LSTM_TFUNC_AMOUNT)
		{
			ret = LSTM_INVALID_ARG;
			goto ERR;
		}
		else
		{
			layerRef[i].inputTFunc = lstm_transfer_list[cfgRef->inputTFunc];
			layerRef[i].inputDTFunc = lstm_transfer_derivative_list[cfgRef->inputTFunc];
		}

		if(cfgRef->outputTFunc < 0 || cfgRef->outputTFunc >= LSTM_TFUNC_AMOUNT)
		{
			ret = LSTM_INVALID_ARG;
			goto ERR;
		}
		else
		{
			layerRef[i].outputTFunc = lstm_transfer_list[cfgRef->outputTFunc];
			layerRef[i].outputDTFunc = lstm_transfer_derivative_list[cfgRef->outputTFunc];
		}

		if(cfgRef->gateTFunc < 0 || cfgRef->gateTFunc >= LSTM_TFUNC_AMOUNT)
		{
			ret = LSTM_INVALID_ARG;
			goto ERR;
		}
		else
		{
			layerRef[i].gateTFunc = lstm_transfer_list[cfgRef->gateTFunc];
			layerRef[i].gateDTFunc = lstm_transfer_derivative_list[cfgRef->gateTFunc];
		}
	}

	// Assign value
	lstm->layerList = layerRef;
	goto RET;

ERR:
	for(i = 0; i < cfgRef->layers; i++)
	{
		lstm_layer_delete(&layerRef[i]);
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_base_alloc(struct LSTM_BASE* basePtr, int netSize, int reNetSize)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_BASE tmpBase;

	LOG("enter");

	// Zero memory
	memset(&tmpBase, 0, sizeof(struct LSTM_BASE));

	// Allocate memory
	if(netSize > 0)
	{
		lstm_alloc(tmpBase.weight, netSize, double, ret, ERR);
		lstm_alloc(tmpBase.wGrad, netSize, double, ret, ERR);
		lstm_alloc(tmpBase.wDelta, netSize, double, ret, ERR);
	}

	if(reNetSize > 0)
	{
		lstm_alloc(tmpBase.rWeight, reNetSize, double, ret, ERR);
		lstm_alloc(tmpBase.rGrad, reNetSize, double, ret, ERR);
		lstm_alloc(tmpBase.rDelta, reNetSize, double, ret, ERR);
	}

	// Assign value
	*basePtr = tmpBase;
	goto RET;

ERR:
	lstm_base_delete(&tmpBase);

RET:
	LOG("exit");
	return ret;
}

int lstm_node_alloc(struct LSTM_NODE* nodePtr, int nodeType, int netSize, int reNetSize)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_NODE tmpNode;

	LOG("enter");

	// Zero memory
	memset(&tmpNode, 0, sizeof(struct LSTM_NODE));

	// Allocate base
	switch(nodeType)
	{
		case LSTM_FULL_NODE:
			// Allocate gate networks
			ret = lstm_base_alloc(&tmpNode.ogNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

			ret = lstm_base_alloc(&tmpNode.fgNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

			ret = lstm_base_alloc(&tmpNode.igNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

		case LSTM_OUTPUT_NODE:
			// Allocate input network
			ret = lstm_base_alloc(&tmpNode.inputNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

		case LSTM_INPUT_NODE:
			break;

		default:
			ret = LSTM_INVALID_ARG;
			goto RET;
	}

	// Assign value
	*nodePtr = tmpNode;
	goto RET;

ERR:
	lstm_node_delete(&tmpNode);

RET:
	LOG("exit");
	return ret;
}

int lstm_layer_alloc(struct LSTM_LAYER* layerPtr, int nodeCount, int nodeType, int netSize, int reNetSize)
{
	int i;
	int ret = LSTM_NO_ERROR;
	struct LSTM_LAYER tmpLayer;

	LOG("enter");

	// Zero memory
	memset(&tmpLayer, 0, sizeof(struct LSTM_LAYER));

	// Allocate node list
	lstm_alloc(tmpLayer.nodeList, nodeCount, struct LSTM_NODE, ret, ERR);

	// Allocate nodes
	for(i = 0; i < nodeCount; i++)
	{
		ret = lstm_node_alloc(&tmpLayer.nodeList[i], nodeType, netSize, reNetSize);
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}
	}

	// Assign value
	*layerPtr = tmpLayer;
	goto RET;

ERR:
	lstm_layer_delete(&tmpLayer);

RET:
	LOG("exit");
	return ret;
}

