#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	int i;
	int iResult;
	lstm_config_t cfg, cpy;

	iResult = lstm_config_create(&cfg);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Set config
	lstm_config_set_inputs(cfg, 1);
	lstm_config_set_outputs(cfg, 2);
	iResult = lstm_config_set_hidden_layers(cfg, 3);
	for(i = 0; i < 3; i++)
	{
		lstm_config_set_hidden_nodes(cfg, i, 4 + i);
	}
	lstm_config_set_input_transfer_func(cfg, 7);
	lstm_config_set_output_transfer_func(cfg, 8);

	// Print detail of config
	printf("Src config:\n");
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

	// Clone config
	iResult = lstm_config_clone(&cpy, cfg);
	if(iResult < 0)
	{
		printf("lstm_config_clone() failed with error: %d\n", iResult);
		return -1;
	}

	printf("\n");

	// Print detail of config
	printf("Cpy config:\n");
	printf("Inputs: %d\n", cpy->inputs);
	printf("Outputs: %d\n", cpy->outputs);
	printf("Layers: %d\n", cpy->layers);
	printf("\n");
	printf("Input tfunc index: %d\n", cpy->inputTFunc);
	printf("Output tfunc index: %d\n", cpy->outputTFunc);
	printf("Gate tfunc index: %d\n", cpy->gateTFunc);
	printf("\n");

	for(i = 0; i < cpy->layers; i++)
	{
		printf("Nodes of %d layer: %d\n", i, cpy->nodeList[i]);
	}

	// Cleanup
	lstm_config_delete(cfg);
	lstm_config_delete(cpy);

	return 0;
}

