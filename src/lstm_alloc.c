#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

int lstm_base_alloc(struct LSTM_BASE* basePtr, int netSize, int reNetSize)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_BASE tmpBase;

	LOG("enter");

	// Zero memory
	memset(&tmpBase, 0, sizeof(struct LSTM_BASE));

	// Allocate memory
	if(netSize > 0)
	{
		lstm_alloc(tmpBase.weight, netSize, double, ret, ERR);
		lstm_alloc(tmpBase.wGrad, netSize, double, ret, ERR);
		lstm_alloc(tmpBase.wDelta, netSize, double, ret, ERR);
	}

	if(reNetSize > 0)
	{
		lstm_alloc(tmpBase.rWeight, reNetSize, double, ret, ERR);
		lstm_alloc(tmpBase.rGrad, reNetSize, double, ret, ERR);
		lstm_alloc(tmpBase.rDelta, reNetSize, double, ret, ERR);
	}

	// Assign value
	*basePtr = tmpBase;
	goto RET;

ERR:
	lstm_base_delete(&tmpBase);

RET:
	LOG("exit");
	return ret;
}

int lstm_node_alloc(struct LSTM_NODE* nodePtr, int nodeType, int netSize, int reNetSize)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_NODE tmpNode;

	LOG("enter");

	// Zero memory
	memset(&tmpNode, 0, sizeof(struct LSTM_NODE));

	// Allocate base
	switch(nodeType)
	{
		case LSTM_FULL_NODE:
			// Allocate gate networks
			ret = lstm_base_alloc(&tmpNode.ogNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

			ret = lstm_base_alloc(&tmpNode.fgNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

			ret = lstm_base_alloc(&tmpNode.igNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}

		case LSTM_OUTPUT_NODE:
			// Allocate input network
			ret = lstm_base_alloc(&tmpNode.inputNet, netSize, reNetSize);
			if(ret != LSTM_NO_ERROR)
			{
				goto ERR;
			}
			break;

		default:
			ret = LSTM_INVALID_ARG;
			goto RET;
	}

	// Assign value
	*nodePtr = tmpNode;
	goto RET;

ERR:
	lstm_node_delete(&tmpNode);

RET:
	LOG("exit");
	return ret;
}

