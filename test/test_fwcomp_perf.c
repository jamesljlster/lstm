#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

#include <lstm.h>
#include <lstm_print.h>

#define FW_ITER	100000

int main(int argc, char* argv[])
{
	int i, iter, hSize;
	int iResult;
	int inputs, outputs;

	lstm_t lstm;
	lstm_config_t cfg;

	float* input = NULL;
	float* output = NULL;
	char* tmpPtr;

	clock_t timeHold;

	srand(time(NULL));

	// Checking
	if(argc < 3)
	{
		printf("Usage: test_fwcomp_perf <hidden_nodes> <iteration>\n");
		return -1;
	}

	// Parsing argument
	hSize = strtol(argv[1], &tmpPtr, 10);
	if(tmpPtr == argv[1])
	{
		printf("Failed to parse %s as hidden nodes!\n", tmpPtr);
		return -1;
	}

	iter = strtol(argv[2], &tmpPtr, 10);
	if(tmpPtr == argv[2])
	{
		printf("Failed to parse %s as iteration!\n", tmpPtr);
		return -1;
	}

	printf("Using hidden nodes: %d\n", hSize);
	printf("Using iteration: %d\n", iter);

	// Create lstm config
	iResult = lstm_config_create(&cfg);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Set hidden size
	iResult = lstm_config_set_hidden_nodes(cfg, 0, hSize);
	if(iResult < 0)
	{
		printf("lstm_config_set_hidden_nodes() failed with error: %d\n", iResult);
		return -1;
	}

	// Create lstm
	iResult = lstm_create(&lstm, cfg);
	if(iResult < 0)
	{
		printf("lstm_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Get neural network configuration
	inputs = lstm_config_get_inputs(cfg);
	outputs = lstm_config_get_outputs(cfg);

	// Memory allocation
	input = calloc(inputs, sizeof(float));
	output = calloc(outputs, sizeof(float));
	if(input == NULL || output == NULL)
	{
		printf("Memory allocation failed\n");
		return -1;
	}

	timeHold = clock();
	for(i = 0; i < iter; i++)
	{
		lstm_forward_computation(lstm, input, output);
	}
	timeHold = clock() - timeHold;

	// Print summary
	printf("=== LSTM Arch ===\n");
	printf("    Inputs:  %d\n", lstm_config_get_inputs(cfg));
	printf("    Outputs: %d\n", lstm_config_get_outputs(cfg));
	printf("    Hidden Layers: %d\n", lstm_config_get_hidden_layers(cfg));
	for(i = 0; i < lstm_config_get_hidden_layers(cfg); i++)
	{
		printf("    Hidden Nodes of Hidden Layer %d: %d\n", i, lstm_config_get_hidden_nodes(cfg, i));
	}
	printf("\n");

	printf("Time cost of %d iteration forward computation: %lf sec\n", FW_ITER, (float)timeHold / (float)CLOCKS_PER_SEC);

	lstm_delete(lstm);
	lstm_config_delete(cfg);

	return 0;
}
