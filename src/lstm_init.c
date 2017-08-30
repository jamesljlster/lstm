#include <stdlib.h>
#include <time.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

#define NUM_PRECISION	1000
#define NUM_MAX			1
#define	NUM_MIN			-1

double lstm_rand(void)
{
	int randRange;

	randRange = (NUM_MAX - NUM_MIN) * NUM_PRECISION + 1;

	return (double)(rand() % randRange) / (double)(NUM_PRECISION) + (double)NUM_MIN;
}

double lstm_zero(void)
{
	return 0;
}

void lstm_init_network(lstm_t lstm, double (*initMethod)(void))
{
	int i, j, k;
	struct LSTM_LAYER* layerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	LOG("enter");

	srand((unsigned int)time(NULL));

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
				layerRef[i].nodeList[j].inputNet.weight[k] = initMethod();
				if(i < cfgRef->layers - 1)
				{
					layerRef[i].nodeList[j].ogNet.weight[k] = initMethod();
					layerRef[i].nodeList[j].fgNet.weight[k] = initMethod();
					layerRef[i].nodeList[j].igNet.weight[k] = initMethod();
				}
			}

			// Rand recurrent weight
			if(i == 1)
			{
				for(k = 0; k < layerRef[cfgRef->layers - 2].nodeCount; k++)
				{
					layerRef[i].nodeList[j].inputNet.rWeight[k] = initMethod();
					layerRef[i].nodeList[j].ogNet.rWeight[k] = initMethod();
					layerRef[i].nodeList[j].fgNet.rWeight[k] = initMethod();
					layerRef[i].nodeList[j].igNet.rWeight[k] = initMethod();
				}
			}

			// Rand threshold
			layerRef[i].nodeList[j].inputNet.th = initMethod();
			if(i < cfgRef->layers - 1)
			{
				layerRef[i].nodeList[j].ogNet.th = initMethod();
				layerRef[i].nodeList[j].fgNet.th = initMethod();
				layerRef[i].nodeList[j].igNet.th = initMethod();
			}
		}
	}

	LOG("exit");
}

void lstm_rand_network(lstm_t lstm)
{
	lstm_init_network(lstm, lstm_rand);
}

void lstm_zero_network(lstm_t lstm)
{
	lstm_init_network(lstm, lstm_zero);
}

