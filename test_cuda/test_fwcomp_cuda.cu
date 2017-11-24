#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lstm.h>
#include <lstm_cuda.h>
#include <lstm_print.h>

int main(int argc, char* argv[])
{
	int i;
	int iResult;
	int inputs, outputs;

	cudaError_t cuErr;

	lstm_t lstm;
	lstm_cuda_t lstmCuda;
	lstm_config_t cfg;

	double* input = NULL;
	double* output = NULL;

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

	// Clone lstm to cuda
	iResult = lstm_clone_to_cuda(&lstmCuda, lstm);
	if(iResult < 0)
	{
		printf("lstm_clone_to_cuda() failed with error: %d\n", iResult);
		return -1;
	}

	// Get neural network configuration
	cfg = lstm_get_config(lstm);
	inputs = lstm_config_get_inputs(cfg);
	outputs = lstm_config_get_outputs(cfg);

	lstm_fprint_config(stdout, cfg, 0);
	lstm_fprint_net(stdout, lstm, 0);

	// Memory allocation
	input = (double*)calloc(inputs, sizeof(double));
	output = (double*)calloc(outputs, sizeof(double));
	if(input == NULL || output == NULL)
	{
		printf("Memory allocation failed\n");
		return -1;
	}

	while(1)
	{
		for(i = 0; i < inputs; i++)
		{
			printf("Assign %d of %d input: ", i + 1, inputs);
			iResult = scanf(" %lf", &input[i]);
			if(iResult <= 0)
			{
				i--;
				continue;
			}
		}

		lstm_forward_computation(lstm, input, output);
		for(i = 0; i < outputs; i++)
		{
			printf("%d of %d output: %lf\n", i + 1, outputs, output[i]);
		}

		memset(output, 0, outputs * sizeof(double));

		lstm_forward_computation_cuda(lstmCuda, input, output);
		for(i = 0; i < outputs; i++)
		{
			printf("%d of %d output (with cuda): %lf\n", i + 1, outputs, output[i]);
		}
		cuErr = cudaGetLastError();
		printf("Last cuda error: %s, %s\n", cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));

		printf("\n");
	}

	lstm_delete(lstm);
	lstm_delete_cuda(lstmCuda);

	return 0;
}
