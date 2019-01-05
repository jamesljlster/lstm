#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	int i;
	int iResult;
	lstm_config_t cfg1, cfg2, cpy;

	iResult = lstm_config_create(&cfg1);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	iResult = lstm_config_create(&cfg2);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Print detail of config 2
	printf("Src config:\n");
	printf("Inputs: %d\n", cfg2->inputs);
	printf("Outputs: %d\n", cfg2->outputs);
	printf("Layers: %d\n", cfg2->layers);
	printf("\n");
	printf("Input tfunc index: %d\n", cfg2->inputTFunc);
	printf("Output tfunc index: %d\n", cfg2->outputTFunc);
	printf("Gate tfunc index: %d\n", cfg2->gateTFunc);
	printf("\n");

	for(i = 0; i < cfg2->layers; i++)
	{
		printf("Nodes of %d layer: %d\n", i, cfg2->nodeList[i]);
	}

	// Set config
	lstm_config_set_inputs(cfg1, 1);
	lstm_config_set_outputs(cfg1, 2);
	iResult = lstm_config_set_hidden_layers(cfg1, 3);
	for(i = 0; i < 3; i++)
	{
		lstm_config_set_hidden_nodes(cfg1, i, 4 + i);
	}
	lstm_config_set_input_transfer_func(cfg1, 7);
	lstm_config_set_output_transfer_func(cfg1, 8);

	// Print detail of config
	printf("Src config:\n");
	printf("Inputs: %d\n", cfg1->inputs);
	printf("Outputs: %d\n", cfg1->outputs);
	printf("Layers: %d\n", cfg1->layers);
	printf("\n");
	printf("Input tfunc index: %d\n", cfg1->inputTFunc);
	printf("Output tfunc index: %d\n", cfg1->outputTFunc);
	printf("Gate tfunc index: %d\n", cfg1->gateTFunc);
	printf("\n");

	for(i = 0; i < cfg1->layers; i++)
	{
		printf("Nodes of %d layer: %d\n", i, cfg1->nodeList[i]);
	}

	// Clone config
	iResult = lstm_config_clone(&cpy, cfg1);
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

	printf("\n");

	// Test compare
	iResult = lstm_config_arch_compare(cfg1, cfg2);
	printf("Compare result of cfg1 and cfg1: %d\n", iResult);

	iResult = lstm_config_arch_compare(cfg1, cpy);
	printf("Compare result of cfg1 and cpy: %d\n", iResult);

	iResult = lstm_config_arch_compare(cfg2, cpy);
	printf("Compare result of cfg2 and cpy: %d\n", iResult);

	// Cleanup
	lstm_config_delete(cfg1);
	lstm_config_delete(cpy);

	return 0;
}

