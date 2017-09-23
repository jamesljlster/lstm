#include <stdio.h>

#include <lstm_cuda.h>
#include <lstm_private_cuda.h>

int main()
{
	int i;
	int iResult;

	lstm_cuda_t lstmCuda;
	lstm_config_t cfg;

	iResult = lstm_config_create(&cfg);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Print detail of config
	printf("Inputs: %d\n", cfg->inputs);
	printf("Outputs: %d\n", cfg->outputs);
	printf("Layers: %d\n", cfg->layers);
	printf("\n");
	printf("Input tfunc index: %d\n", cfg->inputTFunc);
	printf("Output tfunc index: %d\n", cfg->outputTFunc);
	printf("Gate tfunc index: %d\n", cfg->gateTFunc);
	printf("\n");

	for(i = 0; i < cfg->layers; i++)
	{
		printf("Nodes of %d layer: %d\n", i, cfg->nodeList[i]);
	}

	// Create lstm cuda
	iResult = lstm_create_cuda(&lstmCuda, cfg);
	if(iResult < 0)
	{
		printf("lstm_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Cleanup
	lstm_config_delete(cfg);
	lstm_delete_cuda(lstmCuda);

	return 0;
}

