
#include <cuda_runtime.h>

#include <debug.h>
#include "lstm_private_cuda.h"

void lstm_forward_computation_erase_cuda(lstm_cuda_t lstmCuda)
{
}

__global__ void lstm_matmul(double* outBuf, double* input, double* weight)
{
}

int lstm_forward_computation_cuda(lstm_cuda_t lstmCuda, double* input, double* output)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int indexTmp;

	struct LSTM_CONFIG_STRUCT* cfgRef;
	struct LSTM_CULAYER* layerRef;

	LOG("enter");

	// Set referenct
	cfgRef = &lstmCuda->config;
	layerRef = lstmCuda->layerList;

	// Copy inputs
	cudaMemcpy(layerRef[0].out, input,
			cfgRef->inputs * sizeof(double),
			cudaMemcpyHostToDevice);

	// Copy recurrent output
	indexTmp = cfgRef->layers - 2;
	cudaMemcpy(&layerRef[0].out[cfgRef->inputs], layerRef[indexTmp].output,
			layerRef[indexTmp].nodeCount * sizeof(double),
			cudaMemcpyDeviceToDevice);

	// Hidden layer calculation
	for(i = 1; i < cfgRef->layers - 1; i++)
	{
	}

	// Output layer calculation
	indexTmp = cfgRef->layers - 1;

	// Get output
	if(output != NULL)
	{
		cudaMemcpy(output, layerRef[indexTmp].out, cfgRef->outputs * sizeof(double));
	}

RET:
	LOG("exit");
	return ret;
}

