#include <stdio.h>
#include <stdlib.h>

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

	iResult = lstm_export(lstm, "test_export.lstm");
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_export() failed with error: %d\n", iResult);
	}

	lstm_delete(lstm);

	return 0;
}
