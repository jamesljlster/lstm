#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

#define INPUTS	2
#define OUTPUTS	1
#define H_LAYERS	1
#define H_NODES		2

#define TIME_STEP 2
#define DATA_COLS 3

extern double dataset[];

int main()
{
	int i, j, k, row;
	int iResult;

	lstm_t lstm;
	lstm_config_t cfg;

	double input[INPUTS];
	double output[OUTPUTS];
	double err[OUTPUTS];

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

	// Set time step
	iResult = lstm_bptt_set_max_timestep(lstm, TIME_STEP);
	if(iResult < 0)
	{
		printf("lstm_bptt_set_max_timestep() failed with error: %d\n", iResult);
		return -1;
	}

	// Sum gradient
	for(row = 0; row < TIME_STEP; row++)
	{
		for(i = 0; i < INPUTS; i++)
		{
			input[i] = dataset[row * DATA_COLS + i];
		}

		// Forward computation
		lstm_forward_computation(lstm, input, output);

		// Print detail
		for(i = 0; i < H_LAYERS; i++)
		{
			printf("=== Hidden layer %d ===\n", i);
			for(j = 0; j < H_NODES; j++)
			{
				printf("Hidden node %d:\n", j);
				printf("    [Input net]\n");
				printf("        calc: %lf\n", lstm->layerList[i + 1].nodeList[j].inputNet.calc);
				printf("        out:  %lf\n", lstm->layerList[i + 1].nodeList[j].inputNet.out);
				printf("\n");
				printf("    [Input gate]\n");
				printf("        calc: %lf\n", lstm->layerList[i + 1].nodeList[j].igNet.calc);
				printf("        out:  %lf\n", lstm->layerList[i + 1].nodeList[j].igNet.out);
				printf("\n");
				printf("    [Forget gate]\n");
				printf("        calc: %lf\n", lstm->layerList[i + 1].nodeList[j].fgNet.calc);
				printf("        out:  %lf\n", lstm->layerList[i + 1].nodeList[j].fgNet.out);
				printf("\n");
				printf("    [Output gate]\n");
				printf("        calc: %lf\n", lstm->layerList[i + 1].nodeList[j].ogNet.calc);
				printf("        out:  %lf\n", lstm->layerList[i + 1].nodeList[j].ogNet.out);
				printf("\n");
				printf("    cell:   %lf\n", lstm->layerList[i + 1].nodeList[j].cell);
				printf("    output: %lf\n", lstm->layerList[i + 1].nodeList[j].output);
				printf("\n");
			}
		}

		for(i = 0; i < INPUTS; i++)
		{
			printf("%d of %d input: %lf\n", i + 1, INPUTS, input[i]);
		}

		for(i = 0; i < OUTPUTS; i++)
		{
			printf("%d of %d output: %lf\n", i + 1, OUTPUTS, output[i]);
		}

		for(i = 0; i < OUTPUTS; i++)
		{
			err[i] = dataset[row * DATA_COLS + INPUTS + i] - output[i];
			printf("%d of %d error: %lf\n", i + 1, OUTPUTS, err[i]);
		}
		printf("\n");

		// Backpropagation
		lstm_bptt_sum_gradient(lstm, err);

		// Print bp detail
		printf("=== BP: Output layer ===\n");
		for(j = 0; j < OUTPUTS; j++)
		{
			printf("Output node: %d\n", j);
			printf("    grad:   %lf\n", lstm->layerList[cfg->layers - 1].nodeList[j].inputNet.grad);
			printf("    thGrad: %lf\n", lstm->layerList[cfg->layers - 1].nodeList[j].inputNet.thGrad);
			printf("    wGrad:  ");
			for(k = 0; k < lstm->layerList[cfg->layers - 2].nodeCount; k++)
			{
				printf("%lf, ", lstm->layerList[cfg->layers - 1].nodeList[j].inputNet.wGrad[k]);
			}
			printf("\n");
		}
		printf("\n");

		printf("=== BP: Hidden layer ===\n");
		for(i = cfg->layers - 2; i > 0; i--)
		{
			for(j = 0; j < H_NODES; j++)
			{
				printf("Hidden node: %d\n", j);
				printf("    grad:  %lf\n", lstm->layerList[i].nodeList[j].grad);
				printf("\n");

				printf("    [Output gate]\n");
				printf("        grad:   %lf\n", lstm->layerList[i].nodeList[j].ogNet.grad);
				printf("        thGrad: %lf\n", lstm->layerList[i].nodeList[j].ogNet.thGrad);
				printf("        wGrad:  ");
				for(k = 0; k < lstm->layerList[i - 1].nodeCount; k++)
				{
					printf("%lf, ", lstm->layerList[i].nodeList[j].ogNet.wGrad[k]);
				}
				printf("\n");
				if(i == 1)
				{
					printf("        rGrad:  ");
					for(k = 0; k < lstm->layerList[cfg->layers - 2].nodeCount; k++)
					{
						printf("%lf, ", lstm->layerList[i].nodeList[j].ogNet.rGrad[k]);
					}
					printf("\n");
				}
				printf("\n");

				printf("    [Forget gate]\n");
				printf("        grad:   %lf\n", lstm->layerList[i].nodeList[j].fgNet.grad);
				printf("        thGrad: %lf\n", lstm->layerList[i].nodeList[j].fgNet.thGrad);
				printf("        wGrad:  ");
				for(k = 0; k < lstm->layerList[i - 1].nodeCount; k++)
				{
					printf("%lf, ", lstm->layerList[i].nodeList[j].fgNet.wGrad[k]);
				}
				printf("\n");
				if(i == 1)
				{
					printf("        rGrad:  ");
					for(k = 0; k < lstm->layerList[cfg->layers - 2].nodeCount; k++)
					{
						printf("%lf, ", lstm->layerList[i].nodeList[j].fgNet.rGrad[k]);
					}
					printf("\n");
				}
				printf("\n");

				printf("    [Input gate]\n");
				printf("        grad:   %lf\n", lstm->layerList[i].nodeList[j].igNet.grad);
				printf("        thGrad: %lf\n", lstm->layerList[i].nodeList[j].igNet.thGrad);
				printf("        wGrad:  ");
				for(k = 0; k < lstm->layerList[i - 1].nodeCount; k++)
				{
					printf("%lf, ", lstm->layerList[i].nodeList[j].igNet.wGrad[k]);
				}
				printf("\n");
				if(i == 1)
				{
					printf("        rGrad:  ");
					for(k = 0; k < lstm->layerList[cfg->layers - 2].nodeCount; k++)
					{
						printf("%lf, ", lstm->layerList[i].nodeList[j].igNet.rGrad[k]);
					}
					printf("\n");
				}
				printf("\n");

				printf("    [Input net]\n");
				printf("        grad:   %lf\n", lstm->layerList[i].nodeList[j].inputNet.grad);
				printf("        thGrad: %lf\n", lstm->layerList[i].nodeList[j].inputNet.thGrad);
				printf("        wGrad:  ");
				for(k = 0; k < lstm->layerList[i - 1].nodeCount; k++)
				{
					printf("%lf, ", lstm->layerList[i].nodeList[j].inputNet.wGrad[k]);
				}
				printf("\n");
				if(i == 1)
				{
					printf("        rGrad:  ");
					for(k = 0; k < lstm->layerList[cfg->layers - 2].nodeCount; k++)
					{
						printf("%lf, ", lstm->layerList[i].nodeList[j].inputNet.rGrad[k]);
					}
					printf("\n");
				}
				printf("\n");

			}
		}
	}

	// Cleanup
	lstm_config_delete(cfg);
	lstm_delete(lstm);

	return 0;
}

double dataset[] = {
	0.4, 0.3, 0.5,
	1.0, 0.1, 0.6,
	0.2, 0.6, 0.4
};

