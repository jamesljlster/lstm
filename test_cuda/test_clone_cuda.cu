#include <stdio.h>

#include <lstm_cuda.h>
#include <lstm_private_cuda.h>

int main(int argc, char* argv[])
{
	int i, j;
	int iResult, ret = 0;

	cudaError_t cuErr;

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

/*
	// Clone lstm from cuda
	iResult = lstm_clone_from_cuda(&cmpLstm, cuLstm);
	if(iResult < 0)
	{
		printf("lstm_clone_from_cuda() failed with error: %d\n", iResult);
		ret = -1;
		goto RET;
	}
	*/

RET:
	cuErr = cudaGetLastError();
	printf("Last cuda error: %s, %s\n", cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));
	/*
	lstm_delete(srcLstm);
	lstm_delete(cmpLstm);
	lstm_delete_cuda(cuLstm);
	*/

	return 0;
}
