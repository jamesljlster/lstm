
#include <cuda_runtime.h>

#include <debug.h>
#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

//#define DUMP_CUDA
#include "lstm_dump_cuda.h"

void lstm_forward_computation_erase_cuda(lstm_cuda_t lstmCuda)
{
	int i, indexTmp;
	struct LSTM_CULAYER* layerRef;

	LOG("enter");

	// Set referenct
	layerRef = lstmCuda->layerList;

	// Set indexTmp to last hidden layer index
	indexTmp = lstmCuda->config.layers - 2;

	// Clear hidden layer outputs
	cudaMemset(layerRef[indexTmp].output, 0, layerRef[indexTmp].nodeCount * sizeof(double));

	// Clear cell value
	for(i = 1; i <= indexTmp; i++)
	{
		cudaMemset(layerRef[i].cell, 0, layerRef[i].nodeCount * sizeof(double));
	}

	LOG("exit");
}

// lstm_matmul<<<LSTM_CUMAT_AMOUNT * layerRef[i].nodeCount, layerRef[i].vecLen - 1>>>
__global__ void lstm_matmul(double* calcBuf, double* input, double* weight, int vecLen)
{
	calcBuf[blockIdx.x * vecLen + threadIdx.x] =
		weight[blockIdx.x * vecLen + threadIdx.x] * input[threadIdx.x];
}

// lstm_base_calc<<<LSTM_CUMAT_AMOUNT, layerRef[i].nodeCount>>>
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
	out[blockIdx.x * blockDim.x + threadIdx.x] = 0;
	tFuncIndex = (blockIdx.x == LSTM_CUMAT_INPUT) ? inputTFunc : gateTFunc;
	lstm_transfer_list_cu[tFuncIndex](&out[blockIdx.x * blockDim.x + threadIdx.x],
			calc[blockIdx.x * blockDim.x + threadIdx.x]);
}

// lstm_cell_calc<<<1, layerRef[i].nodeCount>>>
__global__ void lstm_cell_calc(double* output, double* cell, double* baseOut, int outputTFunc)
{
	double calcTmp;

	// Find cell value
	cell[threadIdx.x] = baseOut[LSTM_CUMAT_FG * blockDim.x + threadIdx.x] * cell[threadIdx.x] +
		baseOut[LSTM_CUMAT_IG * blockDim.x + threadIdx.x] *
		baseOut[LSTM_CUMAT_INPUT * blockDim.x + threadIdx.x];

	// Find output value
	lstm_transfer_list_cu[outputTFunc](&calcTmp, cell[threadIdx.x]);
	output[threadIdx.x] = baseOut[LSTM_CUMAT_OG * blockDim.x + threadIdx.x] * calcTmp;
}

void lstm_forward_computation_cuda(lstm_cuda_t lstmCuda, double* input, double* output)
{
	int i;
	int indexTmp;

	struct LSTM_CONFIG_STRUCT* cfgRef;
	struct LSTM_CULAYER* layerRef;

	LOG("enter");

	// Set referenct
	cfgRef = &lstmCuda->config;
	layerRef = lstmCuda->layerList;

	// Copy inputs
	cudaMemcpy(layerRef[0].output, input,
			cfgRef->inputs * sizeof(double),
			cudaMemcpyHostToDevice);

	// Copy recurrent output
	indexTmp = cfgRef->layers - 2;
	cudaMemcpy(&layerRef[0].output[cfgRef->inputs], layerRef[indexTmp].output,
			layerRef[indexTmp].nodeCount * sizeof(double),
			cudaMemcpyDeviceToDevice);

	DUMP("layerRef[0].output: ", layerRef[0].output, layerRef[0].nodeCount);

	// Hidden layer calculation
	for(i = 1; i < cfgRef->layers - 1; i++)
	{
		// Matrix element multiplication
		lstm_matmul<<<LSTM_CUMAT_AMOUNT * layerRef[i].nodeCount, layerRef[i].vecLen - 1>>>(
				layerRef[i].baseMat.calcBuf,
				layerRef[i - 1].output,
				layerRef[i].baseMat.weight,
				layerRef[i].vecLen);
		//cudaDeviceSynchronize();

		lstm_base_calc<<<LSTM_CUMAT_AMOUNT, layerRef[i].nodeCount>>>(
				layerRef[i].baseMat.calc,
				layerRef[i].baseMat.out,
				layerRef[i].baseMat.calcBuf,
				layerRef[i].baseMat.weight,
				layerRef[i].vecLen,
				layerRef[i].inputTFunc,
				layerRef[i].gateTFunc);
		//cudaDeviceSynchronize();

		// LSTM cell calculation
		lstm_cell_calc<<<1, layerRef[i].nodeCount>>>(
				layerRef[i].output,
				layerRef[i].cell,
				layerRef[i].baseMat.out,
				layerRef[i].outputTFunc);
		//cudaDeviceSynchronize();
	}

	// Output layer calculation
	indexTmp = cfgRef->layers - 1;
	lstm_matmul<<<layerRef[indexTmp].nodeCount, layerRef[indexTmp].vecLen - 1>>>(
			layerRef[indexTmp].baseMat.calcBuf,
			layerRef[indexTmp - 1].output,
			layerRef[indexTmp].baseMat.weight,
			layerRef[indexTmp].vecLen);
	//cudaDeviceSynchronize();

	lstm_base_calc<<<1, layerRef[indexTmp].nodeCount>>>(
			layerRef[indexTmp].baseMat.calc,
			layerRef[indexTmp].baseMat.out,
			layerRef[indexTmp].baseMat.calcBuf,
			layerRef[indexTmp].baseMat.weight,
			layerRef[indexTmp].vecLen,
			layerRef[indexTmp].inputTFunc,
			layerRef[indexTmp].gateTFunc);
	//cudaDeviceSynchronize();

	// Get output
	if(output != NULL)
	{
		cudaMemcpy(output, layerRef[indexTmp].baseMat.out, cfgRef->outputs * sizeof(double), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaDeviceSynchronize();
	}

	LOG("exit");
}

