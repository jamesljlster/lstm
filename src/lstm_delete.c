#include <stdlib.h>
#include <string.h>

#include "lstm_private.h"

#include "debug.h"

void lstm_delete(lstm_t lstm)
{
	LOG("enter");

	lstm_struct_delete(lstm);
	lstm_free(lstm);

	LOG("exit");
}

void lstm_buf_delete(struct LSTM_BUF* bufPtr)
{
	LOG("enter");

	// Free memory
	lstm_free(bufPtr->list);

	// Zero memory
	memset(bufPtr, 0, sizeof(struct LSTM_BUF));

	LOG("exit");
}

void lstm_base_delete(struct LSTM_BASE* basePtr)
{
	LOG("enter");

	// Free memory
	lstm_free(basePtr->weight);
	lstm_free(basePtr->wGrad);
	lstm_free(basePtr->wDelta);

	lstm_free(basePtr->rWeight);
	lstm_free(basePtr->rGrad);
	lstm_free(basePtr->rDelta);

	// Zero memory
	memset(basePtr, 0, sizeof(struct LSTM_BASE));

	LOG("exit");
}

void lstm_node_delete(struct LSTM_NODE* nodePtr)
{
	LOG("enter");

	// Delete base structures
	lstm_base_delete(&nodePtr->ogNet);
	lstm_base_delete(&nodePtr->fgNet);
	lstm_base_delete(&nodePtr->igNet);
	lstm_base_delete(&nodePtr->inputNet);

	// Delete queue buffers
	lstm_buf_delete(&nodePtr->outputQue);
	lstm_buf_delete(&nodePtr->cellQue);
	lstm_buf_delete(&nodePtr->ogQue);
	lstm_buf_delete(&nodePtr->ogCalcQue);
	lstm_buf_delete(&nodePtr->fgQue);
	lstm_buf_delete(&nodePtr->fgCalcQue);
	lstm_buf_delete(&nodePtr->igQue);
	lstm_buf_delete(&nodePtr->igCalcQue);

	// Zero memory
	memset(nodePtr, 0, sizeof(struct LSTM_NODE));

	LOG("exit");
}

void lstm_layer_delete(struct LSTM_LAYER* layerPtr)
{
	int i;

	LOG("enter");

	// Delete node list
	for(i = 0; i < layerPtr->nodeCount; i++)
	{
		lstm_node_delete(&layerPtr->nodeList[i]);
	}

	// Zero memory
	memset(layerPtr, 0, sizeof(struct LSTM_LAYER));

	LOG("exit");
}

void lstm_struct_delete(struct LSTM_STRUCT* lstm)
{
	int i;

	LOG("enter");

	// Delete layers
	for(i = 0; i < lstm->config.layers; i++)
	{
		lstm_layer_delete(&lstm->layerList[i]);
	}

	// Delete config
	lstm_config_struct_delete(&lstm->config);

	// Zero memory
	memset(lstm, 0, sizeof(struct LSTM_STRUCT));

	LOG("exit");
}

