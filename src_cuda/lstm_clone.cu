
#include <cuda_runtime.h>

#include <debug.h>

#include "lstm_cuda.h"
#include "lstm_private_cuda.h"

int lstm_clone_to_cuda(lstm_cuda_t* lstmCudaPtr, lstm_t lstm)
{
	int i, j;
	int ret = LSTM_NO_ERROR;

	lstm_cuda_t tmpLstmCuda = NULL;
	struct LSTM_CULAYER* dstLayerRef;
	struct LSTM_LAYER* srcLayerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	LOG("enter");

	// Get reference
	cfgRef = &lstm->config;

	// Create lstm cuda
	lstm_run(lstm_create_cuda(&tmpLstmCuda, cfgRef), ret, RET);

	// Clone weight
	dstLayerRef = tmpLstmCuda->layerList;
	srcLayerRef = lstm->layerList;
	for(i = 1; i < cfgRef->layers; i++)
	{
		for(j = 0; j < srcLayerRef[i].nodeCount; j++)
		{
			if(i < cfgRef->layers - 1)
			{
				// Clone output gate network
				// Clone forget gate network
				// Clone input gate network
			}

			// Clone input network
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_clone_form_cuda(lstm_t* lstmPtr, lstm_cuda_t lstmCuda)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

RET:
	LOG("exit");
	return ret;
}
