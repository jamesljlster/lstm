#include <string.h>

#include <cuda_runtime.h>

#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

#include <debug.h>

int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize)
{
	int i;
	int matDim, vecLen;
	int ret = LSTM_NO_ERROR;
	struct LSTM_CULAYER tmpLayer;

	LOG("enter");

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
	vecLen = netSize + reNetSize;

	// Allocate device memory
	lstm_alloc_cuda(tmpLayer.nodeMat.weight, matDim * nodeCount * vecLen, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.wGrad, matDim * nodeCount * vecLen, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.wDelta, matDim * nodeCount * vecLen, double, ret, ERR);

	lstm_alloc_cuda(tmpLayer.nodeMat.calcBuf, matDim * nodeCount * vecLen, double, ret, ERR);

	lstm_alloc_cuda(tmpLayer.nodeMat.th, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.thGrad, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.thDelta, matDim * nodeCount, double, ret, ERR);

	lstm_alloc_cuda(tmpLayer.nodeMat.calc, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.out, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.grad, matDim * nodeCount, double, ret, ERR);
	lstm_alloc_cuda(tmpLayer.nodeMat.gradHold, matDim * nodeCount, double, ret, ERR);

	// Assign value
	*cuLayerPtr = tmpLayer;
	goto RET;

ERR:
	lstm_layer_delete_cuda(&tmpLayer);

RET:
	LOG("exit");
	return ret;
}

