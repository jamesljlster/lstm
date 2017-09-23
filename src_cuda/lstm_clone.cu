
#include <cuda_runtime.h>

#include <debug.h>

#include "lstm_cuda.h"
#include "lstm_private_cuda.h"

int lstm_clone_to_cuda(lstm_cuda_t* lstmCudaPtr, lstm_t lstm)
{
	int i, j;
	int vecLen, nodeCount;
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

#define __lstm_clone_base_to_cuda(cuMat, base) \
	{ \
		int procIndex = cuMat * (nodeCount * vecLen) + j * vecLen; \
		if(cudaMemcpy(&dstLayerRef[i].nodeMat.weight[procIndex], \
					srcLayerRef[i].nodeList[j].base.weight, \
					srcLayerRef[i - 1].nodeCount * sizeof(double), \
					cudaMemcpyHostToDevice) \
				!= cudaSuccess) \
		{ \
			ret = LSTM_MEM_FAILED; \
			goto ERR; \
		} \
		else \
		{ \
			procIndex += srcLayerRef[i - 1].nodeCount; \
		} \
 \
		if(i == 1) \
		{ \
			if(cudaMemcpy(&dstLayerRef[i].nodeMat.weight[procIndex], \
						srcLayerRef[i].nodeList[j].base.rWeight, \
						srcLayerRef[cfgRef->layers - 2].nodeCount * sizeof(double), \
						cudaMemcpyHostToDevice) \
					!= cudaSuccess) \
			{ \
				ret = LSTM_MEM_FAILED; \
				goto ERR; \
			} \
			else \
			{ \
				procIndex += srcLayerRef[cfgRef->layers - 2].nodeCount; \
			} \
		} \
 \
		if(cudaMemcpy(&dstLayerRef[i].nodeMat.weight[procIndex], \
					&srcLayerRef[i].nodeList[j].base.th, \
					sizeof(double), \
					cudaMemcpyHostToDevice) \
				!= cudaSuccess) \
		{ \
			ret = LSTM_MEM_FAILED; \
			goto ERR; \
		} \
	}

	// Clone weight
	dstLayerRef = tmpLstmCuda->layerList;
	srcLayerRef = lstm->layerList;
	for(i = 1; i < cfgRef->layers; i++)
	{
		// Get node count and vector length
		nodeCount = srcLayerRef[i].nodeCount;
		vecLen = dstLayerRef[i].vecLen;

		// Clone node to cuda
		for(j = 0; j < nodeCount; j++)
		{
			// Clone gate network
			if(i < cfgRef->layers - 1)
			{
				__lstm_clone_base_to_cuda(LSTM_CUMAT_OG, ogNet);
				__lstm_clone_base_to_cuda(LSTM_CUMAT_FG, fgNet);
				__lstm_clone_base_to_cuda(LSTM_CUMAT_IG, igNet);
			}

			// Clone input network
			__lstm_clone_base_to_cuda(LSTM_CUMAT_INPUT, inputNet);
		}
	}

	// Assign value
	*lstmCudaPtr = tmpLstmCuda;

	goto RET;

ERR:
	lstm_delete_cuda(tmpLstmCuda);

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
