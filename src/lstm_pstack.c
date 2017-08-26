#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_pstack.h"

#include "debug.h"

void lstm_pstack_init(struct LSTM_PSTACK* pstack)
{
	LOG("enter");

	pstack->stack = NULL;
	pstack->stackSize = 0;
	pstack->top = -1;

	LOG("exit");
}

int lstm_pstack_push(struct LSTM_PSTACK* pstack, void* ptr)
{
	int ret = LSTM_NO_ERROR;
	void* allocTmp;

	LOG("enter");

	if(pstack->top == pstack->stackSize - 1)
	{
		allocTmp = realloc(pstack->stack, sizeof(void*) * (pstack->stackSize + 1));
		if(allocTmp == NULL)
		{
			ret = LSTM_MEM_FAILED;
			goto RET;
		}
		else
		{
			pstack->stackSize++;
			pstack->stack = allocTmp;
			allocTmp = NULL;
		}
	}

	pstack->top++;
	pstack->stack[pstack->top] = ptr;

RET:
	LOG("exit");
	return ret;
}

void* lstm_pstack_pop(struct LSTM_PSTACK* pstack)
{
	void* retPtr;

	LOG("enter");

	if(pstack->top < 0)
	{
		retPtr = NULL;
	}
	else
	{
		retPtr = pstack->stack[pstack->top];
		pstack->top--;
	}

	LOG("exit");
	return retPtr;
}

void lstm_pstack_delete(struct LSTM_PSTACK* pstack)
{
	LOG("enter");

	lstm_free(pstack->stack);
	lstm_pstack_init(pstack);

	LOG("exit");
}
