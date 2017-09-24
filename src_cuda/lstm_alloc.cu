#include <string.h>

#include <cuda_runtime.h>

#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

#include <debug.h>

int lstm_network_alloc_cuda(struct LSTM_CUDA* lstmCuda, const struct LSTM_CONFIG_STRUCT* lstmCfg)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int nodeType, netSize, reNetSize, nodeCount;

	int layers;
	int* nodeList;

	void* allocTmp;
	struct LSTM_CULAYER* layerRef = NULL;

	LOG("enter");

	// Set reference
	layers = lstmCfg->layers;
	nodeList = lstmCfg->nodeList;

	// Checking
	if(layers < 3 || nodeList == NULL)
	{
		ret = LSTM_INVALID_ARG;
		goto RET;
	}

	// Allocate layer list
	lstm_alloc(allocTmp, layers, struct LSTM_CULAYER, ret, RET);
	layerRef = (struct LSTM_CULAYER*)allocTmp;

	for(i = 0; i < layers; i++)
	{
		// Set node type, recurrent network size and node count
		if(i == 0)
		{
			nodeType = LSTM_INPUT_NODE;
			reNetSize = 0;
			netSize = 0;
			nodeCount = nodeList[i] + nodeList[layers - 2];
		}
		else if(i == 1)
		{
			nodeType = LSTM_FULL_NODE;
			reNetSize = nodeList[layers - 2];
			netSize = nodeList[i - 1];
			nodeCount = nodeList[i];
		}
		else if(i == layers - 1)
		{
			nodeType = LSTM_OUTPUT_NODE;
			reNetSize = 0;
			netSize = nodeList[i - 1];
			nodeCount = nodeList[i];
		}
		else
		{
			nodeType = LSTM_FULL_NODE;
			reNetSize = 0;
			netSize = nodeList[i - 1];
			nodeCount = nodeList[i - 1];
		}

		// Allocate layer struct
		lstm_run(lstm_layer_alloc_cuda(&layerRef[i], nodeCount, nodeType, netSize, reNetSize), ret, ERR);

		// Set activation function
		layerRef[i].inputTFunc = lstmCfg->inputTFunc;
		layerRef[i].outputTFunc = lstmCfg->outputTFunc;
		layerRef[i].gateTFunc = lstmCfg->gateTFunc;
	}

	// Assing value
	lstmCuda->layerList = layerRef;

	goto RET;

ERR:
	if(layerRef != NULL)
	{
		for(i = 0; i < layers; i++)
		{
			lstm_layer_delete_cuda(&layerRef[i]);
		}
		lstm_free(layerRef);
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize)
{
	int i;
	int matDim, vecLen;
	int ret = LSTM_NO_ERROR;
	struct LSTM_CULAYER tmpLayer;

	LOG("enter");

	// Checking
	if(nodeCount <= 0)
	{
		ret = LSTM_INVALID_ARG;
		goto RET;
	}

	// Zero memory
	memset(&tmpLayer, 0, sizeof(struct LSTM_CULAYER));

	// Find matrix dimension
	matDim = 0;
	switch(nodeType)
	{
		case LSTM_FULL_NODE:
			matDim = LSTM_CUMAT_AMOUNT;
			break;

		case LSTM_OUTPUT_NODE:
		case LSTM_INPUT_NODE:
			matDim = 1;
			break;

		default:
			ret = LSTM_INVALID_ARG;
			goto RET;
	}

	// Find vector length
	vecLen = netSize + reNetSize + 1;

	// Allocate device memory for network base
	if(vecLen > 1)
	{
		lstm_alloc_cuda(tmpLayer.baseMat.weight, matDim * nodeCount * vecLen, double, ret, ERR);
		lstm_alloc_cuda(tmpLayer.baseMat.wGrad, matDim * nodeCount * vecLen, double, ret, ERR);
		lstm_alloc_cuda(tmpLayer.baseMat.wDelta, matDim * nodeCount * vecLen, double, ret, ERR);

		lstm_alloc_cuda(tmpLayer.baseMat.calcBuf, matDim * nodeCount * vecLen, double, ret, ERR);
	}

	lstm_alloc_cuda(tmpLayer.baseMat.calc, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.baseMat.out, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.baseMat.grad, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.baseMat.gradHold, matDim * nodeCount, double, ret, ERR);

	// Allocate device memory for lstm block
	lstm_alloc_cuda(tmpLayer.output, nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.cell, nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.grad, nodeCount, double, ret, ERR);

	// Set value
	tmpLayer.vecLen = vecLen;
	tmpLayer.nodeCount = nodeCount;

	// Assign value
	*cuLayerPtr = tmpLayer;
	goto RET;

ERR:
	lstm_layer_delete_cuda(&tmpLayer);

RET:
	LOG("exit");
	return ret;
}

