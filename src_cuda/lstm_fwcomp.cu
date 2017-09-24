
#include <cuda_runtime.h>

#include <debug.h>
#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

void lstm_forward_computation_erase_cuda(lstm_cuda_t lstmCuda)
{
}

__global__ void lstm_matmul(double* calcBuf, double* input, double* weight, int vecLen)
{
	calcBuf[blockIdx.x * vecLen + threadIdx.x] =
		weight[blockIdx.x * vecLen + threadIdx.x] * input[threadIdx.x];
}

__global__ void lstm_base_calc(double* calc, double* out, double* calcBuf, double* weight, int vecLen, int inputTFunc, int gateTFunc)
{
	int i;
	int tFuncIndex;

	// Copy threshold value to calcBuf
	calcBuf[blockIdx.x * (blockDim.x * vecLen) + (threadIdx.x + 1) * vecLen - 1] =
		weight[blockIdx.x * (blockDim.x * vecLen) + (threadIdx.x + 1) * vecLen - 1];

	// Vector reduction
	calc[blockIdx.x * blockDim.x + threadIdx.x] = 0;
	for(i = 0; i < vecLen; i++)
	{
		calc[blockIdx.x * blockDim.x + threadIdx.x] +=
			calcBuf[blockIdx.x * (blockDim.x * vecLen) + threadIdx.x * vecLen + i];
	}

	// Find network base output
	tFuncIndex = (blockIdx.x == LSTM_CUMAT_INPUT) ? inputTFunc : gateTFunc;
	lstm_transfer_list_cu[tFuncIndex](&out[blockIdx.x * blockDim.x + threadIdx.x],
			calc[blockIdx.x * blockDim.x + threadIdx.x]);
}

int lstm_forward_computation_cuda(lstm_cuda_t lstmCuda, double* input, double* output)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int indexTmp;
	int vecCalcLen;

	struct LSTM_CONFIG_STRUCT* cfgRef;
	struct LSTM_CULAYER* layerRef;

	LOG("enter");

	// Set referenct
	cfgRef = &lstmCuda->config;
	layerRef = lstmCuda->layerList;

	// Copy inputs
	cudaMemcpy(layerRef[0].nodeMat.out, input,
			cfgRef->inputs * sizeof(double),
			cudaMemcpyHostToDevice);

	// Copy recurrent output
	indexTmp = cfgRef->layers - 2;
	cudaMemcpy(&layerRef[0].nodeMat.out[cfgRef->inputs], layerRef[indexTmp].nodeMat.out,
			layerRef[indexTmp].nodeCount * sizeof(double),
			cudaMemcpyDeviceToDevice);

	// Hidden layer calculation
	for(i = 1; i < cfgRef->layers - 1; i++)
	{
		// Matrix element multiplication
		lstm_matmul<<<LSTM_CUMAT_AMOUNT * layerRef[i].nodeCount, layerRef[i].vecLen - 1>>>(
				layerRef[i].nodeMat.calcBuf,
				layerRef[i - 1].nodeMat.out,
				layerRef[i].nodeMat.weight,
				layerRef[i].vecLen);

		// Network base calculation
		lstm_base_calc<<<LSTM_CUMAT_AMOUNT, layerRef[i].nodeCount>>>(
				layerRef[i].nodeMat.calc,
				layerRef[i].nodeMat.out,
				layerRef[i].nodeMat.calcBuf,
				layerRef[i].nodeMat.weight,
				layerRef[i].vecLen,
				layerRef[i].inputTFunc,
				layerRef[i].gateTFunc);

		// LSTM cell calculation
	}

	// Output layer calculation
	indexTmp = cfgRef->layers - 1;

	// Get output
	if(output != NULL)
	{
		cudaMemcpy(output, layerRef[indexTmp].nodeMat.out, cfgRef->outputs * sizeof(double), cudaMemcpyDeviceToHost);
	}

RET:
	LOG("exit");
	return ret;
}

