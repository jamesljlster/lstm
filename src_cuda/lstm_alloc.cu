#include <string.h>

#include <cuda_runtime.h>

#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

#include <debug.h>

int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize)
{
	int i;
	int ret = LSTM_NO_ERROR;
	struct LSTM_CULAYER tmpLayer;

	LOG("enter");

	// Zero memory
	memset(&tmpLayer, 0, sizeof(struct LSTM_CULAYER));

RET:
	LOG("exit");
	return ret;
}
