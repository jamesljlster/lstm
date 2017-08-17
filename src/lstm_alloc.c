#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm.h"
#include "lstm_private.h"

#include "debug.h"

#define lstm_alloc(ptr, len, type, retVar, errLabel) \
	ptr = calloc(len, sizeof(type)); \
	if(ptr == NULL) \
	{ \
		ret = LSTM_MEM_FAILED; \
		goto errLabel; \
	}

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

