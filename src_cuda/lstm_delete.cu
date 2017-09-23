#include <string.h>
#include <cuda_runtime.h>

#include "lstm_private_cuda.h"

#include <debug.h>

void lstm_delete_cuda(lstm_cuda_t lstmCuda)
{
	LOG("enter");

	if(lstmCuda != NULL)
	{
		lstm_struct_delete_cuda(lstmCuda);
		lstm_free(lstmCuda);
	}

	LOG("exit");
}

void lstm_struct_delete_cuda(struct LSTM_CUDA* lstmCuda)
{
	int i;

	LOG("enter");

	// Delete layers
	if(lstmCuda->layerList != NULL)
	{
		for(i = 0; i < lstmCuda->config.layers; i++)
		{
			lstm_layer_delete_cuda(&lstmCuda->layerList[i]);
		}
		lstm_free(lstmCuda->layerList);
	}

	// Delete config
	lstm_config_struct_delete(&lstmCuda->config);

	// Zero memory
	memset(lstmCuda, 0, sizeof(struct LSTM_CUDA));

	LOG("exit");
}

void lstm_layer_delete_cuda(struct LSTM_CULAYER* cuLayerPtr)
{
	LOG("enter");

	// Free device memory
	lstm_free_cuda(cuLayerPtr->nodeMat.weight);
	lstm_free_cuda(cuLayerPtr->nodeMat.wGrad);
	lstm_free_cuda(cuLayerPtr->nodeMat.wDelta);

	lstm_free_cuda(cuLayerPtr->nodeMat.calcBuf);

	lstm_free_cuda(cuLayerPtr->nodeMat.calc);
	lstm_free_cuda(cuLayerPtr->nodeMat.out);
	lstm_free_cuda(cuLayerPtr->nodeMat.grad);
	lstm_free_cuda(cuLayerPtr->nodeMat.gradHold);

	lstm_free_cuda(cuLayerPtr->nodeMat.outQue);
	lstm_free_cuda(cuLayerPtr->nodeMat.calcQue);

	// Zero memory
	memset(cuLayerPtr, 0, sizeof(struct LSTM_CULAYER));

	LOG("exit");
}

