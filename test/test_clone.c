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
	lstm_t clone;
	lstm_config_t cfg;

	float* input = NULL;
	float* output = NULL;

	// Checking
	if(argc <= 1)
	{
		printf("Assign a lstm model file to run the program\n");
		return -1;
	}

	// Import
	iResult = lstm_import(&lstm, argv[1]);
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_import() failed with error: %d\n", iResult);
		return -1;
	}

	// Clone
	iResult = lstm_clone(&clone, lstm);
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_clone() failed with error: %d\n", iResult);
		return -1;
	}

	// Export
	iResult = lstm_export(clone, "lstm.clone");
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_export() failed with error: %d\n", iResult);
		return -1;
	}

	lstm_delete(lstm);
	lstm_delete(clone);

	return 0;
}
