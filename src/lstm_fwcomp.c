
#include "lstm_private.h"

#include "debug.h"

void lstm_forward_computation_erase(lstm_t lstm)
{
	int i, j;
	int indexTmp;

	struct LSTM_LAYER* layerRef;

	LOG("enter");

	// Set reference
	layerRef = lstm->layerList;

	// Set indexTmp to last hidden layer index
	indexTmp = lstm->config.layers - 2;

	// Clear hidden layer outputs
	for(i = 0; i < layerRef[indexTmp].nodeCount; i++)
	{
		layerRef[indexTmp].nodeList[i].output = 0;
	}

	// Clear cell value
	for(i = 1; i <= indexTmp; i++)
	{
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			layerRef[i].nodeList[j].cell = 0;
		}
	}

	LOG("exit");
}

void lstm_forward_computation(lstm_t lstm, float* input, float* output)
{
	int i, j, k;
	int indexTmp;
	float calcTmp, ig, fg, og;

	struct LSTM_CONFIG_STRUCT* cfgRef;
	struct LSTM_LAYER* layerRef;

	LOG("enter");

	// Set reference
	cfgRef = &lstm->config;
	layerRef = lstm->layerList;

	// Copy inputs
	for(i = 0; i < cfgRef->inputs; i++)
	{
		layerRef[0].nodeList[i].output = input[i];
	}

	// Backup recurrent output
	indexTmp = cfgRef->layers - 2; // Set indexTmp to last hidden layer
	for(i = 0; i < layerRef[indexTmp].nodeCount; i++)
	{
		layerRef[indexTmp].nodeList[i].rHold = layerRef[indexTmp].nodeList[i].output;
	}

#define __lstm_fwcomp_base(layerIndex, baseName, tfuncName) \
	for(j = 0; j < layerRef[layerIndex].nodeCount; j++) \
	{ \
		calcTmp = 0; \
		if(layerIndex == 1) \
		{ \
			for(k = 0; k < layerRef[indexTmp].nodeCount; k++) \
			{ \
				calcTmp += layerRef[indexTmp].nodeList[k].rHold * \
					layerRef[1].nodeList[j].baseName.rWeight[k]; \
			} \
		} \
		for(k = 0; k < layerRef[layerIndex - 1].nodeCount; k++) \
		{ \
			calcTmp += layerRef[layerIndex - 1].nodeList[k].output * \
				layerRef[layerIndex].nodeList[j].baseName.weight[k]; \
		} \
		calcTmp += layerRef[layerIndex].nodeList[j].baseName.th; \
		layerRef[layerIndex].nodeList[j].baseName.calc = calcTmp; \
		layerRef[layerIndex].nodeList[j].baseName.out = \
			layerRef[layerIndex].tfuncName(calcTmp); \
	}

	// Hidden layer calculation
	for(i = 1; i < cfgRef->layers - 1; i++)
	{
		// Processing base network
		__lstm_fwcomp_base(i, ogNet, gateTFunc);
		__lstm_fwcomp_base(i, fgNet, gateTFunc);
		__lstm_fwcomp_base(i, igNet, gateTFunc);
		__lstm_fwcomp_base(i, inputNet, inputTFunc);

		// Find layer outputs
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			// Copy gate values
			ig = layerRef[i].nodeList[j].igNet.out;
			fg = layerRef[i].nodeList[j].fgNet.out;
			og = layerRef[i].nodeList[j].ogNet.out;

			// Find cell value
			layerRef[i].nodeList[j].cell = fg * layerRef[i].nodeList[j].cell +
				ig * layerRef[i].nodeList[j].inputNet.out;

			// Find output value
			layerRef[i].nodeList[j].output = og *
				layerRef[i].outputTFunc(layerRef[i].nodeList[j].cell);
		}
	}

	// Output layer calculation
	indexTmp = cfgRef->layers - 1; // Set indexTmp to output layer
	for(j = 0; j < layerRef[indexTmp].nodeCount; j++)
	{
		calcTmp = 0;
		for(k = 0; k < layerRef[indexTmp - 1].nodeCount; k++)
		{
			calcTmp += layerRef[indexTmp - 1].nodeList[k].output *
				layerRef[indexTmp].nodeList[j].inputNet.weight[k];
		}

		calcTmp += layerRef[indexTmp].nodeList[j].inputNet.th;

		layerRef[indexTmp].nodeList[j].inputNet.calc = calcTmp;
		layerRef[indexTmp].nodeList[j].inputNet.out = layerRef[indexTmp].outputTFunc(calcTmp);
	}

	// Get output
	if(output != NULL)
	{
		for(i = 0; i < cfgRef->outputs; i++)
		{
			output[i] = layerRef[indexTmp].nodeList[i].inputNet.out;
		}
	}

	LOG("exit");
}

