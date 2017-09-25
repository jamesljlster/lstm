#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

#include <lstm.h>
#include <lstm_cuda.h>
#include <lstm_print.h>

#define FW_ITER	100000

int main(int argc, char* argv[])
{
	int i, iter;
	int iResult;
	int inputs, outputs;

	lstm_t lstm;
	lstm_cuda_t lstmCuda;
	lstm_config_t cfg;

	double* input = NULL;
	double* output = NULL;

	clock_t timeHold;

	srand(time(NULL));

	// Checking
	if(argc <= 1)
	{
		printf("Assign a lstm model file to run the program\n");
		return -1;
	}

	iResult = lstm_import(&lstm, argv[1]);
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_import() failed with error: %d\n", iResult);
		return -1;
	}

	iResult = lstm_clone_to_cuda(&lstmCuda, lstm);
	if(iResult < 0)
	{
		printf("lstm_clone_to_cuda() failed with error: %d\n", iResult);
		return -1;
	}

	// Get neural network configuration
	cfg = lstm_get_config(lstm);

	// Memory allocation
	input = (double*)calloc(inputs, sizeof(double));
	output = (double*)calloc(outputs, sizeof(double));
	if(input == NULL || output == NULL)
	{
		printf("Memory allocation failed\n");
		return -1;
	}

	timeHold = clock();
	for(iter = 0; iter < FW_ITER; iter++)
	{
		lstm_forward_computation_cuda(lstmCuda, input, output);
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

	printf("Time cost of %d iteration forward computation: %lf sec\n", FW_ITER, (double)timeHold / (double)CLOCKS_PER_SEC);

	lstm_delete(lstm);

	return 0;
}
