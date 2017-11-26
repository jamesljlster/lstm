#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_print.h>

int main(int argc, char* argv[])
{
	int i;
	int counter;
	int iResult;
	int inputs, outputs;

	lstm_t lstm;
	lstm_config_t cfg;
	lstm_state_t state;

	float* input = NULL;
	float* output = NULL;

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

	// Create lstm state
	iResult = lstm_state_create(&state, cfg);
	if(iResult < 0)
	{
		printf("lstm_state_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Memory allocation
	input = calloc(inputs, sizeof(float));
	output = calloc(outputs, sizeof(float));
	if(input == NULL || output == NULL)
	{
		printf("Memory allocation failed\n");
		return -1;
	}

	counter = 0;
	printf("Backup and restore lstm state after initialization\n");
	while(1)
	{
		// Restore lstm state
		if(counter > 0)
		{
			lstm_state_restore(state, lstm);
		}

		for(i = 0; i < inputs; i++)
		{
			printf("Assign %d of %d input: ", i + 1, inputs);
			iResult = scanf(" %f", &input[i]);
			if(iResult <= 0)
			{
				i--;
				continue;
			}
		}

		lstm_forward_computation(lstm, input, output);

		for(i = 0; i < outputs; i++)
		{
			printf("%d of %d output: %f\n", i + 1, outputs, output[i]);
		}
		printf("\n");

		// Save lstm state
		if(counter == 0)
		{
			lstm_state_save(state, lstm);
		}

		counter++;
	}

	lstm_delete(lstm);

	return 0;
}
