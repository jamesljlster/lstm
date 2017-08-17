
#include "lstm_private.h"

#include "debug.h"

void rnn_forward_computation(lstm_t lstm, double* input, double* output)
{
	int i, j, k;
	int indexTmp;
	double calcTmp;

	struct LSTM_CONFIG_STRUCT* cfgRef = &lstm->config;
	struct LSTM_LAYER* layerRef = lstm->layerList;

	LOG("enter");

	// Copy inputs
	for(i = 0; i < cfgRef->inputs; i++)
	{
		layerRef[0].nodeList[i].output = input[i];
	}

	// Backup recurrent output
	indexTmp = cfgRef->layers - 2;
	for(i = 0; i < layerRef[indexTmp].nodeCount; i++)
	{
		layerRef[indexTmp].nodeList[i].rHold = layerRef[indexTmp].nodeList[i].output;
	}

#define __rnn_fwcomp_base(baseName, tfuncName)                                 \
	for(j = 0; j < layerRef[i].nodeCount; j++)                                 \
	{                                                                          \
		calcTmp = 0;                                                           \
                                                                               \
		if(i == 1)                                                             \
		{                                                                      \
			for(k = 0; k < layerRef[indexTmp].nodeCount; k++)                  \
			{                                                                  \
				calcTmp += layerRef[indexTmp].nodeList[k].rHold *              \
					layerRef[1].nodeList[j].baseName.rWeight[k];               \
			}                                                                  \
		}                                                                      \
                                                                               \
		for(k = 0; k < layerRef[i - 1].nodeCount; k++)                         \
		{                                                                      \
			calcTmp += layerRef[i - 1].nodeList[k].output *                    \
				layerRef[i].nodeList[j].baseName.weight[k];                    \
		}                                                                      \
                                                                               \
		calcTmp += layerRef[i].nodeList[j].baseName.th;                        \
                                                                               \
		layerRef[i].nodeList[j].baseName.calc = calcTmp;                       \
		layerRef[i].nodeList[j].baseName.out = layerRef[i].tfuncName(calcTmp); \
	}

	// Calculation
	for(i = 1; i < cfgRef->layers; i++)
	{
		// Processing base network
		__rnn_fwcomp_base(ogNet, gateTFunc);
		__rnn_fwcomp_base(fgNet, gateTFunc);
		__rnn_fwcomp_base(igNet, gateTFunc);
		__rnn_fwcomp_base(inputNet, inputTFunc);

		/*
		// Calculate input network
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			calcTmp = 0;

			// Recurrent factor
			if(i == 1)
			{
				for(k = 0; k < layerRef[indexTmp].nodeCount; k++)
				{
					calcTmp += layerRef[indexTmp].nodeList[k].rHold *
						layerRef[1].nodeList[j].inputNet.rWeight[k];
				}
			}

			// Common factor
			for(k = 0; k < layerRef[i - 1].nodeCount; k++)
			{
				calcTmp += layerRef[i - 1].nodeList[k].output *
					layerRef[i].nodeList[j].inputNet.weight[k];
			}

			// Threshold
			calcTmp += layerRef[i].nodeList[j].inputNet.th;

			layerRef[i].nodeList[j].inputNet.calc = calcTmp;
			layerRef[i].nodeList[j].inputNet.out = layerRef[i].inputTFunc(calcTmp);
		}
		*/

	}

	// Get output
	if(output != NULL)
	{
		for(i = 0; i < cfgRef->outputs; i++)
		{
			output[i] = layerRef[cfgRef->layers - 1].nodeList[i].output;
		}
	}

	LOG("exit");
}

