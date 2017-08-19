#include <stdlib.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

#define __lstm_buf_realloc(buf, len, ret, errLabel) \
	if(buf.listLen < len) \
	{ \
		tmpPtr = realloc(buf.list, len * sizeof(double)); \
		if(tmpPtr == NULL) \
		{ \
			ret = LSTM_MEM_FAILED; \
			goto errLabel; \
		} \
		else \
		{ \
			buf.list = tmpPtr; \
		} \
	}

int lstm_bptt_sum_gradient(lstm_t lstm, double* dError)
{
	int i, j, k, re;
	int ret = LSTM_NO_ERROR;
	int indexTmp;
	double calcTmp;

	struct LSTM_LAYER* layerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;

	double* tmpPtr = NULL;
	int queLen = 0;

	LOG("enter");

	// Get reference
	layerRef = lstm->layerList;
	cfgRef = &lstm->config;

	// Re-Allocate queue space and set value
	queLen = lstm->queueLen + 1;
	for(i = 0; i < cfgRef->layers; i++)
	{
		for(j = 0; j < layerRef[i].nodeCount; j++)
		{
			// Re-Allocate queue space
			__lstm_buf_realloc(layerRef[i].nodeList[j].outputQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].cellQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].ogNet.outQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].ogNet.calcQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].fgNet.outQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].fgNet.calcQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].igNet.outQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].igNet.calcQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].inputNet.outQue, queLen, ret, RET);
			__lstm_buf_realloc(layerRef[i].nodeList[j].inputNet.calcQue, queLen, ret, RET);

			// Store value to queue
			layerRef[i].nodeList[j].outputQue.list[queLen - 1] = layerRef[i].nodeList[j].output;
			layerRef[i].nodeList[j].cellQue.list[queLen - 1] = layerRef[i].nodeList[j].cell;

			layerRef[i].nodeList[j].ogNet.outQue.list[queLen - 1] =
				layerRef[i].nodeList[j].ogNet.out;
			layerRef[i].nodeList[j].ogNet.calcQue.list[queLen - 1] =
				layerRef[i].nodeList[j].ogNet.calc;

			layerRef[i].nodeList[j].fgNet.outQue.list[queLen - 1] =
				layerRef[i].nodeList[j].fgNet.out;
			layerRef[i].nodeList[j].fgNet.calcQue.list[queLen - 1] =
				layerRef[i].nodeList[j].fgNet.calc;

			layerRef[i].nodeList[j].igNet.outQue.list[queLen - 1] =
				layerRef[i].nodeList[j].igNet.out;
			layerRef[i].nodeList[j].igNet.calcQue.list[queLen - 1] =
				layerRef[i].nodeList[j].igNet.calc;

			layerRef[i].nodeList[j].inputNet.outQue.list[queLen - 1] =
				layerRef[i].nodeList[j].inputNet.out;
			layerRef[i].nodeList[j].inputNet.calcQue.list[queLen - 1] =
				layerRef[i].nodeList[j].inputNet.calc;
		}
	}

	// Update queue length
	lstm->queueLen = queLen;

	// Find network adjust gradient: Output layer
	indexTmp = cfgRef->layers - 1;
	for(j = 0; j < layerRef[indexTmp].nodeCount; j++)
	{
		// Find gradient
		layerRef[indexTmp].nodeList[j].inputNet.grad =
			dError[j] * layerRef[indexTmp].outputDTFunc(
						layerRef[indexTmp].nodeList[j].inputNet.calc
					);

		// Find threshold adjust amount
		layerRef[indexTmp].nodeList[j].inputNet.thGrad +=
			layerRef[indexTmp].nodeList[j].inputNet.grad;

		// Find weight adjust amount
		for(k = 0; k < layerRef[indexTmp - 1].nodeCount; k++)
		{
			layerRef[indexTmp].nodeList[j].inputNet.wGrad[k] +=
				layerRef[indexTmp].nodeList[j].inputNet.grad *
				layerRef[indexTmp - 1].nodeList[k].output;
		}
	}

	// Find gradient: Hidden layers
	for(re = queLen - 1; re >= 0; re--)
	{
		if(re == queLen - 1)
		{
			// Backpropagation from output layer
			for(i = cfgRef->layers - 2; i > 0; i--)
			{
				for(j = 0; j < layerRef[i].nodeCount; j++)
				{
					// Node gradient
					calcTmp = 0;
					for(k = 0; k < layerRef[i + 1].nodeCount; k++)
					{
						calcTmp += layerRef[i + 1].nodeList[k].ogNet.grad *
							layerRef[i + 1].nodeList[k].ogNet.weight[j];

						if(i > cfgRef->layers - 2)
						{
							calcTmp += layerRef[i + 1].nodeList[k].fgNet.grad *
								layerRef[i + 1].nodeList[k].fgNet.weight[j];
							calcTmp += layerRef[i + 1].nodeList[k].igNet.grad *
								layerRef[i + 1].nodeList[k].igNet.weight[j];
							calcTmp += layerRef[i + 1].nodeList[k].inputNet.grad *
								layerRef[i + 1].nodeList[k].inputNet.weight[j];
						}
					}
					layerRef[i].nodeList[j].grad = calcTmp;

					// Output gate gradient
					layerRef[i].nodeList[j].ogNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].outputTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].gateDTFunc(layerRef[i].nodeList[j].ogNet.calcQue.list[re]);

					// Forget gate gradient
					if(re > 0)
					{
						layerRef[i].nodeList[j].fgNet.grad = layerRef[i].nodeList[j].grad *
							layerRef[i].nodeList[j].ogNet.outQue.list[re] *
							layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
							layerRef[i].nodeList[j].cellQue.list[re - 1] *
							layerRef[i].gateDTFunc(layerRef[i].nodeList[j].fgNet.calcQue.list[re]);
					}

					// Input gate gradient
					layerRef[i].nodeList[j].igNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].nodeList[j].ogNet.outQue.list[re] *
						layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].nodeList[j].inputNet.outQue.list[re] *
						layerRef[i].gateDTFunc(layerRef[i].nodeList[j].igNet.calcQue.list[re]);

					// Input gradient
					layerRef[i].nodeList[j].inputNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].nodeList[j].ogNet.outQue.list[re] *
						layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].nodeList[j].igNet.outQue.list[re] *
						layerRef[i].inputDTFunc(layerRef[i].nodeList[j].inputNet.calcQue.list[re]);
				}
			}
		}
		else
		{
			// Backup recurrent gradient
			for(j = 0; j < layerRef[1].nodeCount; j++)
			{
				layerRef[1].nodeList[j].ogNet.gradHold = layerRef[1].nodeList[j].ogNet.grad;
				layerRef[1].nodeList[j].fgNet.gradHold = layerRef[1].nodeList[j].fgNet.grad;
				layerRef[1].nodeList[j].igNet.gradHold = layerRef[1].nodeList[j].igNet.grad;
				layerRef[1].nodeList[j].inputNet.gradHold = layerRef[1].nodeList[j].inputNet.grad;
			}

			// Backpropagation form recurrent factor
			for(i = cfgRef->layers - 2; i > 0; i++)
			{
				for(j = 0; j < layerRef[i].nodeCount; j++)
				{
					// Node gradient
					if(i == cfgRef->layers - 2)
					{
						// Recurrent factor
						calcTmp = 0;
						for(k = 0; k < layerRef[1].nodeCount; k++)
						{
							calcTmp += layerRef[1].nodeList[k].ogNet.gradHold *
								layerRef[1].nodeList[k].ogNet.rWeight[j];
							calcTmp += layerRef[1].nodeList[k].fgNet.gradHold *
								layerRef[1].nodeList[k].fgNet.rWeight[j];
							calcTmp += layerRef[1].nodeList[k].igNet.gradHold *
								layerRef[1].nodeList[k].igNet.rWeight[j];
							calcTmp += layerRef[1].nodeList[k].inputNet.gradHold *
								layerRef[1].nodeList[k].inputNet.rWeight[j];
						}
						layerRef[i].nodeList[j].grad = calcTmp;
					}
					else
					{
						// Common factor
						calcTmp = 0;
						for(k = 0; k < layerRef[i + 1].nodeCount; k++)
						{
							calcTmp += layerRef[i + 1].nodeList[k].ogNet.grad *
								layerRef[i + 1].nodeList[k].ogNet.weight[j];
							calcTmp += layerRef[i + 1].nodeList[k].fgNet.grad *
								layerRef[i + 1].nodeList[k].fgNet.weight[j];
							calcTmp += layerRef[i + 1].nodeList[k].igNet.grad *
								layerRef[i + 1].nodeList[k].igNet.weight[j];
							calcTmp += layerRef[i + 1].nodeList[k].inputNet.grad *
								layerRef[i + 1].nodeList[k].inputNet.weight[j];
						}
						layerRef[i].nodeList[j].grad = calcTmp;
					}

					// Output gate gradient
					layerRef[i].nodeList[j].ogNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].outputTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].gateDTFunc(layerRef[i].nodeList[j].ogNet.calcQue.list[re]);

					// Forget gate gradient
					if(re > 0)
					{
						layerRef[i].nodeList[j].fgNet.grad = layerRef[i].nodeList[j].grad *
							layerRef[i].nodeList[j].ogNet.outQue.list[re] *
							layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
							layerRef[i].nodeList[j].cellQue.list[re - 1] *
							layerRef[i].gateDTFunc(layerRef[i].nodeList[j].fgNet.calcQue.list[re]);
					}

					// Input gate gradient
					layerRef[i].nodeList[j].igNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].nodeList[j].ogNet.outQue.list[re] *
						layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].nodeList[j].inputNet.outQue.list[re] *
						layerRef[i].gateDTFunc(layerRef[i].nodeList[j].igNet.calcQue.list[re]);

					// Input gradient
					layerRef[i].nodeList[j].inputNet.grad = layerRef[i].nodeList[j].grad *
						layerRef[i].nodeList[j].ogNet.outQue.list[re] *
						layerRef[i].outputDTFunc(layerRef[i].nodeList[j].cellQue.list[re]) *
						layerRef[i].nodeList[j].igNet.outQue.list[re] *
						layerRef[i].inputDTFunc(layerRef[i].nodeList[j].inputNet.calcQue.list[re]);
				}
			}
		}

		// Sum gradient
		for(i = cfgRef->layers - 2; i > 0; i--)
		{
			for(j = 0; j < layerRef[i].nodeCount; j++)
			{
				// Sum weight gradient
				for(k = 0; k < layerRef[i - 1].nodeCount; k++)
				{
					// Output gate network
					layerRef[i].nodeList[j].ogNet.weight[k] +=
						layerRef[i].nodeList[j].ogNet.grad *
						layerRef[i - 1].nodeList[k].outputQue.list[re];

					// Forget gate network
					layerRef[i].nodeList[j].fgNet.weight[k] +=
						layerRef[i].nodeList[j].fgNet.grad *
						layerRef[i - 1].nodeList[j].outputQue.list[re];

					// Input gate network
					layerRef[i].nodeList[j].igNet.weight[k] +=
						layerRef[i].nodeList[j].igNet.grad *
						layerRef[i - 1].nodeList[j].outputQue.list[re];

					// Input netwrok
					layerRef[i].nodeList[j].inputNet.weight[k] +=
						layerRef[i].nodeList[j].inputNet.grad *
						layerRef[i - 1].nodeList[j].outputQue.list[re];
				}

				// Sum recurrent weight gradient
				if(i == 1 && re > 0)
				{
					indexTmp = cfgRef->layers - 2;
					for(k = 0; k < layerRef[indexTmp].nodeCount; k++)
					{
						// Output gate network
						layerRef[i].nodeList[j].ogNet.rWeight[k] +=
							layerRef[i].nodeList[j].ogNet.grad *
							layerRef[indexTmp].nodeList[k].outputQue.list[re - 1];

						// Forget gate network
						layerRef[i].nodeList[j].fgNet.rWeight[k] +=
							layerRef[i].nodeList[j].fgNet.grad *
							layerRef[indexTmp].nodeList[k].outputQue.list[re - 1];

						// Input gate network
						layerRef[i].nodeList[j].igNet.rWeight[k] +=
							layerRef[i].nodeList[j].igNet.grad *
							layerRef[indexTmp].nodeList[k].outputQue.list[re - 1];

						// Input network
						layerRef[i].nodeList[j].inputNet.rWeight[k] +=
							layerRef[i].nodeList[j].inputNet.grad *
							layerRef[indexTmp].nodeList[k].outputQue.list[re - 1];
					}
				}

				// Sum threshold gradient
				layerRef[i].nodeList[j].ogNet.thGrad += layerRef[i].nodeList[j].ogNet.grad;
				layerRef[i].nodeList[j].igNet.thGrad += layerRef[i].nodeList[j].igNet.grad;
				layerRef[i].nodeList[j].fgNet.thGrad += layerRef[i].nodeList[j].fgNet.grad;
				layerRef[i].nodeList[j].inputNet.thGrad += layerRef[i].nodeList[j].inputNet.grad;
			}
		}
	}

	LOG("exit");

RET:
	return ret;
}

int lstm_bptt_adjust_netwrok(lstm_t lstm, double learningRate, double momentumCoef, double gradLimit);
int lstm_bptt_erase(lstm_t lstm);

