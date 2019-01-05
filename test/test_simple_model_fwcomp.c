#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

#define INPUTS	2
#define OUTPUTS	1
#define H_LAYERS	1
#define H_NODES		2

int main()
{
	int i, j;
	int iResult;

	lstm_t lstm;
	lstm_config_t cfg;

	float input[INPUTS];
	float output[OUTPUTS];

	// Create config
	iResult = lstm_config_create(&cfg);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Set config
	lstm_config_set_inputs(cfg, INPUTS);
	lstm_config_set_outputs(cfg, OUTPUTS);
	lstm_config_set_hidden_layers(cfg, H_LAYERS);
	lstm_config_set_hidden_nodes(cfg, 0, H_NODES);

	// Print detail of config
	printf("Inputs: %d\n", cfg->inputs);
	printf("Outputs: %d\n", cfg->outputs);
	printf("Layers: %d\n", cfg->layers);
	printf("\n");
	printf("Input tfunc index: %d\n", cfg->inputTFunc);
	printf("Output tfunc index: %d\n", cfg->outputTFunc);
	printf("Gate tfunc index: %d\n", cfg->gateTFunc);
	printf("\n");

	for(i = 0; i < cfg->layers; i++)
	{
		printf("Nodes of %d layer: %d\n", i, cfg->nodeList[i]);
	}
	printf("\n");

	// Create lstm
	iResult = lstm_create(&lstm, cfg);
	if(iResult < 0)
	{
		printf("lstm_create() failed with error: %d\n", iResult);
		return -1;
	}

	// Set weight
	lstm->layerList[1].nodeList[0].inputNet.weight[0] = -0.9;
	lstm->layerList[1].nodeList[0].inputNet.weight[1] = -0.8;
	lstm->layerList[1].nodeList[0].inputNet.rWeight[0] = -0.7;
	lstm->layerList[1].nodeList[0].inputNet.rWeight[1] = -0.6;
	lstm->layerList[1].nodeList[0].inputNet.th = 0.1;

	lstm->layerList[1].nodeList[0].igNet.weight[0] = -0.5;
	lstm->layerList[1].nodeList[0].igNet.weight[1] = -0.4;
	lstm->layerList[1].nodeList[0].igNet.rWeight[0] = -0.3;
	lstm->layerList[1].nodeList[0].igNet.rWeight[1] = -0.2;
	lstm->layerList[1].nodeList[0].igNet.th = 0.2;

	lstm->layerList[1].nodeList[0].fgNet.weight[0] = -0.1;
	lstm->layerList[1].nodeList[0].fgNet.weight[1] = 0.1;
	lstm->layerList[1].nodeList[0].fgNet.rWeight[0] = 0.2;
	lstm->layerList[1].nodeList[0].fgNet.rWeight[1] = 0.3;
	lstm->layerList[1].nodeList[0].fgNet.th = 0.3;

	lstm->layerList[1].nodeList[0].ogNet.weight[0] = 0.4;
	lstm->layerList[1].nodeList[0].ogNet.weight[1] = 0.5;
	lstm->layerList[1].nodeList[0].ogNet.rWeight[0] = 0.6;
	lstm->layerList[1].nodeList[0].ogNet.rWeight[1] = 0.7;
	lstm->layerList[1].nodeList[0].ogNet.th = 0.4;

	lstm->layerList[1].nodeList[1].inputNet.weight[0] = 1.0;
	lstm->layerList[1].nodeList[1].inputNet.weight[1] = 0.9;
	lstm->layerList[1].nodeList[1].inputNet.rWeight[0] = 0.8;
	lstm->layerList[1].nodeList[1].inputNet.rWeight[1] = 0.7;
	lstm->layerList[1].nodeList[1].inputNet.th = 0.5;

	lstm->layerList[1].nodeList[1].igNet.weight[0] = 0.6;
	lstm->layerList[1].nodeList[1].igNet.weight[1] = 0.5;
	lstm->layerList[1].nodeList[1].igNet.rWeight[0] = 0.4;
	lstm->layerList[1].nodeList[1].igNet.rWeight[1] = 0.3;
	lstm->layerList[1].nodeList[1].igNet.th = 0.6;

	lstm->layerList[1].nodeList[1].fgNet.weight[0] = 0.2;
	lstm->layerList[1].nodeList[1].fgNet.weight[1] = 0.1;
	lstm->layerList[1].nodeList[1].fgNet.rWeight[0] = -0.1;
	lstm->layerList[1].nodeList[1].fgNet.rWeight[1] = -0.2;
	lstm->layerList[1].nodeList[1].fgNet.th = 0.7;

	lstm->layerList[1].nodeList[1].ogNet.weight[0] = -0.3;
	lstm->layerList[1].nodeList[1].ogNet.weight[1] = -0.4;
	lstm->layerList[1].nodeList[1].ogNet.rWeight[0] = -0.5;
	lstm->layerList[1].nodeList[1].ogNet.rWeight[1] = -0.6;
	lstm->layerList[1].nodeList[1].ogNet.th = 0.8;

	lstm->layerList[2].nodeList[0].inputNet.weight[0] = 0.5;
	lstm->layerList[2].nodeList[0].inputNet.weight[1] = 0.6;
	lstm->layerList[2].nodeList[0].inputNet.th = 0.9;

	// Forward computation
	while(1)
	{
		for(i = 0; i < INPUTS; i++)
		{
			printf("Assign %d of %d input: ", i + 1, INPUTS);
			iResult = scanf(" %f", &input[i]);
			if(iResult <= 0)
			{
				i--;
				continue;
			}
		}

		lstm_forward_computation(lstm, input, output);

		// Print detail
		for(i = 0; i < H_LAYERS; i++)
		{
			printf("=== Hidden layer %d ===\n", i);
			for(j = 0; j < H_NODES; j++)
			{
				printf("Hidden node %d:\n", j);
				printf("    [Input net]\n");
				printf("        calc: %f\n", lstm->layerList[i + 1].nodeList[j].inputNet.calc);
				printf("        out:  %f\n", lstm->layerList[i + 1].nodeList[j].inputNet.out);
				printf("\n");
				printf("    [Input gate]\n");
				printf("        calc: %f\n", lstm->layerList[i + 1].nodeList[j].igNet.calc);
				printf("        out:  %f\n", lstm->layerList[i + 1].nodeList[j].igNet.out);
				printf("\n");
				printf("    [Forget gate]\n");
				printf("        calc: %f\n", lstm->layerList[i + 1].nodeList[j].fgNet.calc);
				printf("        out:  %f\n", lstm->layerList[i + 1].nodeList[j].fgNet.out);
				printf("\n");
				printf("    [Output gate]\n");
				printf("        calc: %f\n", lstm->layerList[i + 1].nodeList[j].ogNet.calc);
				printf("        out:  %f\n", lstm->layerList[i + 1].nodeList[j].ogNet.out);
				printf("\n");
				printf("    cell:   %f\n", lstm->layerList[i + 1].nodeList[j].cell);
				printf("    output: %f\n", lstm->layerList[i + 1].nodeList[j].output);
				printf("\n");
			}
		}

		for(i = 0; i < OUTPUTS; i++)
		{
			printf("%d of %d output: %f\n", i + 1, OUTPUTS, output[i]);
		}
		printf("\n");
	}

	// Cleanup
	lstm_config_delete(cfg);
	lstm_delete(lstm);

	return 0;
}

