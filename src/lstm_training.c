#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

int lstm_training_gradient(lstm_t lstm, double** inputList, double** desireList, double** outputList, double** errList, int timeStep, double gradLimit)
{
	return lstm_training_gradient_custom(lstm, lstm->config.lRate, lstm->config.mCoef, inputList, desireList, outputList, errList, timeStep, gradLimit);
}

int lstm_training_gradient_custom(lstm_t lstm, double lRate, double mCoef, double** inputList, double** desireList, double** outputList, double** errList, int timeStep, double gradLimit)
{
	int i, j;
	int ret = LSTM_NO_ERROR;
	int outputs;

	double* outputStore = NULL;
	double* errorStore = NULL;

	LOG("enter");

	// Get reference
	outputs = lstm->config.outputs;

	// Memory allocation
	lstm_alloc(outputStore, outputs * timeStep, double, ret, RET);
	lstm_alloc(errorStore, outputs * timeStep, double, ret, RET);

	// Recurrent training
	for(i = 0; i < timeStep; i++)
	{
		// Forward computation
		lstm_forward_computation(lstm, inputList[i], &outputStore[i * outputs]);

		// Find error
		for(j = 0; j < outputs; j++)
		{
			errorStore[i * outputs + j] = desireList[i][j] - outputStore[i * outputs + j];
		}

		// Backpropagation
		lstm_run(lstm_bptt_sum_gradient(lstm, &errorStore[i * outputs]), ret, RET);
	}

	// Adjust network
	lstm_bptt_adjust_network(lstm, lRate, mCoef, gradLimit);

	// Erase
	lstm_bptt_erase(lstm);
	lstm_forward_computation_erase(lstm);

	// Copy values
	if(outputList != NULL)
	{
		for(i = 0; i < timeStep; i++)
		{
			for(j = 0; j < outputs; j++)
			{
				outputList[i][j] = outputStore[i * outputs + j];
			}
		}
	}

	if(errList != NULL)
	{
		for(i = 0; i < timeStep; i++)
		{
			for(j = 0; j < outputs; j++)
			{
				errList[i][j] = errorStore[i * outputs + j];
			}
		}
	}

RET:
	lstm_free(outputStore);
	lstm_free(errorStore);

	LOG("exit")
	return ret;
}