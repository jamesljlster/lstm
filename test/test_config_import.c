#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main(int argc, char* argv[])
{
	int i;
	int iResult;
	lstm_config_t cfg;

	if(argc < 2)
	{
		printf("Assign a lstm config xml file to run the program\n");
		return 0;
	}

	iResult = lstm_config_import(&cfg, argv[1]);
	if(iResult < 0)
	{
		printf("lstm_config_import() failed with error: %d\n", iResult);
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

	// Cleanup
	lstm_config_delete(cfg);

	return 0;
}

