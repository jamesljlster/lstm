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
	int i, j;
	int ret = LSTM_NO_ERROR;

	struct LSTM_LAYER* layerRef;
	struct LSTM_CONFIG_STRUCT* cfgRef;
	double* gradHold;

	double* tmpPtr = NULL;
	int queLen = 0;

	LOG("enter");

	// Get reference
	layerRef = lstm->layerList;
	cfgRef = &lstm->config;
	gradHold = lstm->gradHold;

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

	LOG("exit");

RET:
	return ret;
}

int lstm_bptt_adjust_netwrok(lstm_t lstm, double learningRate, double momentumCoef, double gradLimit);
int lstm_bptt_erase(lstm_t lstm);

