#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

int lstm_clone(lstm_t* lstmPtr, lstm_t lstmSrc)
{
	int i, j;
	int layers;
	int ret = LSTM_NO_ERROR;
	lstm_t tmpLstm = NULL;

	struct LSTM_LAYER* srcLayerRef;
	struct LSTM_LAYER* dstLayerRef;

	LOG("enter");

	// Create lstm
	lstm_run(lstm_create(&tmpLstm, &lstmSrc->config), ret, RET);

	// Get reference
	srcLayerRef = lstmSrc->layerList;
	dstLayerRef = tmpLstm->layerList;
	layers = tmpLstm->config.layers;

#define __lstm_vec_clone(link, len) \
	memcpy(dstLayerRef[i].nodeList[j].link, \
			srcLayerRef[i].nodeList[j].link, \
			len * sizeof(float) \
		  );

#define __lstm_value_copy(link) \
	dstLayerRef[i].nodeList[j].link = srcLayerRef[i].nodeList[j].link;

	// Clone network
	for(i = 1; i < layers; i++)
	{
		for(j = 0; j < dstLayerRef[i].nodeCount; j++)
		{
			// Clone weight
			if(i < layers - 1)
			{
				__lstm_vec_clone(ogNet.weight, dstLayerRef[i - 1].nodeCount);
				__lstm_vec_clone(fgNet.weight, dstLayerRef[i - 1].nodeCount);
				__lstm_vec_clone(igNet.weight, dstLayerRef[i - 1].nodeCount);
			}
			__lstm_vec_clone(inputNet.weight, dstLayerRef[i - 1].nodeCount);

			// Clone bias and cell value
			__lstm_value_copy(cell);

			if(i < layers - 1)
			{
				__lstm_value_copy(ogNet.th);
				__lstm_value_copy(igNet.th);
				__lstm_value_copy(fgNet.th);
			}
			__lstm_value_copy(inputNet.th);

			// Clone recurrent weight
			if(i == layers - 2)
			{
				__lstm_vec_clone(ogNet.rWeight, dstLayerRef[1].nodeCount);
				__lstm_vec_clone(fgNet.rWeight, dstLayerRef[1].nodeCount);
				__lstm_vec_clone(igNet.rWeight, dstLayerRef[1].nodeCount);
				__lstm_vec_clone(inputNet.rWeight, dstLayerRef[1].nodeCount);
			}
		}
	}

	// Assign value
	*lstmPtr = tmpLstm;

RET:
	LOG("exit");
	return ret;
}

