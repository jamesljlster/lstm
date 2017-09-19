#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <lstm.h>

#define INPUTS			2
#define OUTPUTS			1
#define HIDDEN_LAYER	1
#define HIDDEN_SIZE		16

#define L_RATE		0.01
#define M_COEF		0.1
#define DECAY		1.0
#define IT_FUNC		LSTM_HYPERBOLIC_TANGENT
#define OT_FUNC		LSTM_HYPERBOLIC_TANGENT

#define STOP_MSE	0.001
#define ITER_COUNT	10000
#define DELTA_LIMIT	30

#define DATA_ROWS	256
#define DATA_COLS	8
#define RAND_SWAP	32768

#define USING_MANUAL_TRAINING
//#define DEBUG

double* adder_dataprep(int rows, int cols);

int main(int argc, char* argv[])
{
	int i, j, k;
	int augendIndex, addendIndex;
	int dataCounter;
	int iResult;
	int iterCount;

	double lRate, mCoef;
	double mse;

	double* inputList[DATA_COLS];
	double* desireList[DATA_COLS];
	double* errList[DATA_COLS];
#ifdef USING_MANUAL_TRAINING
	double* outList[DATA_COLS];
#endif

	clock_t timeHold;

	double* dataset;
	int dataRows, dataCols;

	lstm_t lstm = NULL;
	lstm_config_t cfg = NULL;

	if(argc > 1)
	{
		iResult = lstm_import(&lstm, argv[1]);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("Failed to import neural network\n");
			return -1;
		}
	}
	else
	{
		// Create config
		iResult = lstm_config_create(&cfg);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_config_create() failed with error: %d\n", iResult);
			return -1;
		}

		lstm_config_set_inputs(cfg, INPUTS);
		lstm_config_set_outputs(cfg, OUTPUTS);
		lstm_config_set_input_transfer_func(cfg, IT_FUNC);
		lstm_config_set_output_transfer_func(cfg, OT_FUNC);
		lstm_config_set_learning_rate(cfg, L_RATE);
		lstm_config_set_momentum_coef(cfg, M_COEF);

		iResult = lstm_config_set_hidden_layers(cfg, HIDDEN_LAYER);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_config_set_hidden_layers() failed with error: %d\n", iResult);
			return -1;
		}

		for(i = 0; i < HIDDEN_LAYER; i++)
		{
			iResult = lstm_config_set_hidden_nodes(cfg, i, HIDDEN_SIZE);
			if(iResult != LSTM_NO_ERROR)
			{
				printf("lstm_config_set_nodes() failed with error: %d\n", iResult);
				return -1;
			}
		}

		// Create neural network
		iResult = lstm_create(&lstm, cfg);
		if(iResult != LSTM_NO_ERROR)
		{
			printf("lstm_create() failed with error: %d\n", iResult);
			return -1;
		}

		// Random weight and threshold
		lstm_rand_network(lstm);
	}

	// Prepare dataset
	dataRows = DATA_ROWS;
	dataCols = DATA_COLS;
	dataset = adder_dataprep(dataRows, dataCols);
	if(dataset == NULL)
	{
		printf("Prepare dataset failed!\n");
		return -1;
	}
#ifdef DEBUG
	else
	{
		for(i = 0; i < dataRows; i++)
		{
			for(j = 0; j < dataCols; j++)
			{
				printf("%lf, ", dataset[i * dataCols + j]);
			}
			printf("\n");
		}
	}
	getchar();
#endif

	// Memory allocation
	for(i = 0; i < DATA_COLS; i++)
	{
		inputList[i] = calloc(INPUTS, sizeof(double));
		if(inputList[i] == NULL)
		{
			printf("Memory allocation failed!\n");
			return -1;
		}

		desireList[i] = calloc(OUTPUTS, sizeof(double));
		if(desireList[i] == NULL)
		{
			printf("Memory allocation failed!\n");
			return -1;
		}

		errList[i] = calloc(OUTPUTS, sizeof(double));
		if(errList[i] == NULL)
		{
			printf("Memory allocaton failed!\n");
			return -1;
		}

#ifdef USING_MANUAL_TRAINING
		outList[i] = calloc(OUTPUTS, sizeof(double));
		if(outList[i] == NULL)
		{
			printf("Memory allocation failed!\n");
			return -1;
		}
#endif
	}

#ifdef USING_MANUAL_TRAINING
	iResult = lstm_bptt_set_max_timestep(lstm, DATA_COLS);
	if(iResult < 0)
	{
		printf("lstm_bptt_set_max_timestep() failed with error: %d\n", iResult);
		return -1;
	}
#endif

	// Training
	timeHold = 0;
	iterCount = 0;
	lRate = L_RATE;
	mCoef = M_COEF;
	while(iterCount < ITER_COUNT)
	{
		mse = 0;
		dataCounter = 0;
		for(augendIndex = 0; augendIndex < dataRows - 1; augendIndex++)
		{
			for(addendIndex = augendIndex + 1; addendIndex < dataRows; addendIndex++)
			{
				if(augendIndex + addendIndex < dataRows)
				{
					// Set input and desire list
					for(j = 0; j < DATA_COLS; j++)
					{
						inputList[j][0] = dataset[augendIndex * DATA_COLS + j];
						inputList[j][1] = dataset[addendIndex * DATA_COLS + j];
						desireList[j][0] = dataset[(augendIndex + addendIndex) * DATA_COLS + j];
					}

#ifdef DEBUG
					// Print data
					printf("Input list: \n");
					for(j = DATA_COLS - 1; j >= 0; j--)
					{
						printf("%lf ", inputList[j][0]);
					}
					printf("\n");

					for(j = DATA_COLS - 1; j >= 0; j--)
					{
						printf("%lf ", inputList[j][1]);
					}
					printf("\n");

					printf("Output list: \n");
					for(j = DATA_COLS - 1; j >= 0; j--)
					{
						printf("%lf ", desireList[j][0]);
					}
					printf("\n");
					getchar();
#endif

#ifdef USING_MANUAL_TRAINING
					for(j = 0; j < DATA_COLS; j++)
					{
						lstm_forward_computation(lstm, inputList[j], outList[j]);

						for(k = 0; k < OUTPUTS; k++)
						{
							errList[j][k] = desireList[j][k] - outList[j][k];
						}

						lstm_bptt_sum_gradient(lstm, errList[j]);
					}

					lstm_bptt_adjust_network(lstm, lRate, mCoef, DELTA_LIMIT);

					lstm_bptt_erase(lstm);
					lstm_forward_computation_erase(lstm);
#else
					// Training
					iResult = lstm_training_gradient_custom(lstm, lRate, mCoef, inputList, desireList, NULL, errList, DATA_COLS, DELTA_LIMIT);
					if(iResult != LSTM_NO_ERROR)
					{
						//printf("lstm_training_gradient() failed with error: %s\n", lstm_get_error_msg(iResult));
						printf("lstm_training_gradient() failed with error: %d\n", iResult);
						return -1;
					}
#endif

					// Find error
					for(j = 0; j < DATA_COLS; j++)
					{
						for(k = 0; k < OUTPUTS; k++)
						{
							mse += errList[j][k] * errList[j][k];
						}
					}

					dataCounter++;
				}
			}
		}

		mse /= (double)(DATA_COLS) * (double)(dataCounter) * (double)OUTPUTS;
		printf("Iter. %5d mse: %lf\n", iterCount, mse);

		if(mse <= STOP_MSE)
			break;

		lRate = lRate * DECAY;
		mCoef = mCoef * DECAY;
		iterCount++;
	}

	timeHold = clock() - timeHold;

	printf("\nTime cost: %lf secs\n\n", (double)timeHold / (double)CLOCKS_PER_SEC);

	iResult = lstm_export(lstm, "./test.lstm");
	if(iResult != LSTM_NO_ERROR)
	{
		printf("lstm_export() failed!\n");
	}

	lstm_delete(lstm);
	lstm_config_delete(cfg);

	return 0;
}

double* adder_dataprep(int rows, int cols)
{
	int i, j;

	double* tmpPtr = NULL;

	// Memory allocation
	tmpPtr = calloc(rows * cols, sizeof(double));
	if(tmpPtr == NULL)
	{
		goto RET;
	}

	// Prepare dataset
	for(i = 0; i < rows; i++)
	{
		for(j = 0; j < cols; j++)
		{
			if((i & (1 << j)) > 0)
			{
				tmpPtr[i * cols + j] = 1;
			}
			else
			{
				tmpPtr[i * cols + j] = 0;
			}
		}
	}

RET:
	return tmpPtr;
}

