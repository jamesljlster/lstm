#include <stdio.h>

#include <lstm_cuda.h>
#include <lstm_private_cuda.h>

int main(int argc, char* argv[])
{
	int i, j;
	int iResult, ret = 0;

	cudaError_t cuErr;

	struct LSTM_LAYER* srcLayerRef;
	struct LSTM_LAYER* cmpLayerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	lstm_t srcLstm = NULL;
	lstm_t cmpLstm = NULL;
	lstm_cuda_t cuLstm = NULL;

	// Checking
	if(argc < 2)
	{
		printf("Assing a lstm model to run the program.\n");
		goto RET;
	}

	// Import soruce lstm
	iResult = lstm_import(&srcLstm, argv[1]);
	if(iResult < 0)
	{
		printf("lstm_import() with %s failed with error: %d\n", argv[1], iResult);
		ret = -1;
		goto RET;
	}

	// Clone lstm to cuda
	iResult = lstm_clone_to_cuda(&cuLstm, srcLstm);
	if(iResult < 0)
	{
		printf("lstm_clone_to_cuda() failed with error: %d\n", iResult);
		ret = -1;
		goto RET;
	}

	// Clone lstm from cuda
	iResult = lstm_clone_from_cuda(&cmpLstm, cuLstm);
	if(iResult < 0)
	{
		printf("lstm_clone_from_cuda() failed with error: %d\n", iResult);
		ret = -1;
		goto RET;
	}

#define __lstm_clone_validation(base) \
	iResult = memcmp(srcLayerRef[i].nodeList[j].base.weight, \
			cmpLayerRef[i].nodeList[j].base.weight, \
			srcLayerRef[i - 1].nodeCount * sizeof(double)); \
	if(iResult != 0) \
	{ \
		printf("Clone validation failed!\n"); \
		ret = -1; \
		goto RET; \
	} \
\
	if(i == 1) \
	{ \
		iResult = memcmp(srcLayerRef[i].nodeList[j].base.rWeight, \
				cmpLayerRef[i].nodeList[j].base.rWeight, \
				srcLayerRef[cfgRef->layers - 2].nodeCount * sizeof(double)); \
		if(iResult != 0) \
		{ \
			printf("Clone validation failed!\n"); \
			ret = -1; \
			goto RET; \
		} \
	} \
\
	if(srcLayerRef[i].nodeList[j].base.th != cmpLayerRef[i].nodeList[j].base.th) \
	{ \
		printf("Clone validation failed!\n"); \
		ret = -1; \
		goto RET; \
	} \

	// Verify data
	srcLayerRef = srcLstm->layerList;
	cmpLayerRef = cmpLstm->layerList;
	cfgRef = &srcLstm->config;
	for(i = 1; i < cfgRef->layers; i++)
	{
		for(j = 0; j < srcLayerRef[i].nodeCount; j++)
		{
			if(i < cfgRef->layers - 1)
			{
				__lstm_clone_validation(ogNet);
				__lstm_clone_validation(igNet);
				__lstm_clone_validation(fgNet);
			}

			__lstm_clone_validation(inputNet);
		}
	}

RET:
	cuErr = cudaGetLastError();
	printf("Last cuda error: %s, %s\n", cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));

	lstm_delete(srcLstm);
	lstm_delete(cmpLstm);
	lstm_delete_cuda(cuLstm);

	return 0;
}
