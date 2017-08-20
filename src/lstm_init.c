#include <stdlib.h>
#include <time.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

#define NUM_PRECISION	1000
#define NUM_MAX			1
#define	NUM_MIN			-1

double lstm_rand()
{
	int randRange;

	randRange = (NUM_MAX - NUM_MIN) * NUM_PRECISION + 1;

	return (double)(rand() % randRange) / (double)(NUM_PRECISION) + (double)NUM_MIN;
}

void lstm_rand_network(lstm_t lstm)
{
	int i, j, k;
	struct LSTM_LAYER* layerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	LOG("enter");

	srand(time(NULL));

	// Get reference
	layerRef = lstm->layerList;
	cfgRef = &lstm->config;

	// Rand network
	for(i = 1; i < cfgRef->layers; i++)
	{
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			// Rand weight
			for(k = 0; k < layerRef[i - 1].nodeCount; k++)
			{
				layerRef[i].nodeList[j].inputNet.weight[k] = lstm_rand();
				if(i < cfgRef->layers - 1)
				{
					layerRef[i].nodeList[j].ogNet.weight[k] = lstm_rand();
					layerRef[i].nodeList[j].fgNet.weight[k] = lstm_rand();
					layerRef[i].nodeList[j].igNet.weight[k] = lstm_rand();
				}
			}

			// Rand recurrent weight
			if(i == 1)
			{
				for(k = 0; k < layerRef[cfgRef->layers - 1].nodeCount; k++)
				{
					layerRef[i].nodeList[j].inputNet.rWeight[k] = lstm_rand();
					layerRef[i].nodeList[j].ogNet.rWeight[k] = lstm_rand();
					layerRef[i].nodeList[j].fgNet.rWeight[k] = lstm_rand();
					layerRef[i].nodeList[j].igNet.rWeight[k] = lstm_rand();
				}
			}

			// Rand threshold
			layerRef[i].nodeList[j].inputNet.th = lstm_rand();
			if(i < cfgRef->layers - 1)
			{
				layerRef[i].nodeList[j].ogNet.th = lstm_rand();
				layerRef[i].nodeList[j].fgNet.th = lstm_rand();
				layerRef[i].nodeList[j].igNet.th = lstm_rand();
			}
		}
	}

	LOG("exit");
}

void lstm_zero_network(lstm_t lstm)
{
	int i, j, k;
	struct LSTM_LAYER* layerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	LOG("enter");

	// Get reference
	layerRef = lstm->layerList;
	cfgRef = &lstm->config;

	// Rand network
	for(i = 1; i < cfgRef->layers; i++)
	{
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			// Rand weight
			for(k = 0; k < layerRef[i - 1].nodeCount; k++)
			{
				layerRef[i].nodeList[j].inputNet.weight[k] = 0;
				if(i < cfgRef->layers - 1)
				{
					layerRef[i].nodeList[j].ogNet.weight[k] = 0;
					layerRef[i].nodeList[j].fgNet.weight[k] = 0;
					layerRef[i].nodeList[j].igNet.weight[k] = 0;
				}
			}

			// Rand recurrent weight
			if(i == 1)
			{
				for(k = 0; k < layerRef[cfgRef->layers - 1].nodeCount; k++)
				{
					layerRef[i].nodeList[j].inputNet.rWeight[k] = 0;
					layerRef[i].nodeList[j].ogNet.rWeight[k] = 0;
					layerRef[i].nodeList[j].fgNet.rWeight[k] = 0;
					layerRef[i].nodeList[j].igNet.rWeight[k] = 0;
				}
			}

			// Rand threshold
			layerRef[i].nodeList[j].inputNet.th = 0;
			if(i < cfgRef->layers - 1)
			{
				layerRef[i].nodeList[j].ogNet.th = 0;
				layerRef[i].nodeList[j].fgNet.th = 0;
				layerRef[i].nodeList[j].igNet.th = 0;
			}
		}
	}

	LOG("exit");
}

