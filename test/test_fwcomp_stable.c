#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

#include <lstm.h>
#include <lstm_print.h>

int main(int argc, char* argv[])
{
	int i;
	int iResult;
	int inputs, outputs;

	lstm_t lstm;
	lstm_config_t cfg;

	double* input = NULL;
	double* output = NULL;

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

	// Get neural network configuration
	cfg = lstm_get_config(lstm);
	inputs = lstm_config_get_inputs(cfg);
	outputs = lstm_config_get_outputs(cfg);

	lstm_fprint_config(stdout, cfg, 0);
	lstm_fprint_net(stdout, lstm, 0);

	// Memory allocation
	input = calloc(inputs, sizeof(double));
	output = calloc(outputs, sizeof(double));
	if(input == NULL || output == NULL)
	{
		printf("Memory allocation failed\n");
		return -1;
	}

	while(!kbhit())
	{
		for(i = 0; i < inputs; i++)
		{
			input[i] = (double)(rand()% 2001 - 1000) / 1000.0;
			printf("%d of %d input: %lf\n", i + 1, inputs, input[i]);
		}

		lstm_forward_computation(lstm, input, output);

		for(i = 0; i < outputs; i++)
		{
			printf("%d of %d output: %lf\n", i + 1, outputs, output[i]);
		}
		printf("\n");
	}

	lstm_delete(lstm);

	return 0;
}
