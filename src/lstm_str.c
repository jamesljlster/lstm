#include <stdlib.h>

#include "lstm.h"
#include "lstm_str.h"

#include "debug.h"

int lstm_str_append(struct LSTM_STR* strPtr, char ch)
{
	int ret = LSTM_NO_ERROR;
	int tmpLen;

	void* allocTmp;

	LOG("enter");

	// Find new memory length
	tmpLen = strPtr->strLen + 2;

	// Reallocate memory
	if(tmpLen > strPtr->memLen)
	{
		allocTmp = realloc(strPtr->str, tmpLen * sizeof(char));
		if(allocTmp == NULL)
		{
			ret = LSTM_MEM_FAILED;
			goto RET;
		}
		else
		{
			strPtr->str = allocTmp;
		}
	}

	// Append character
	strPtr->str[strPtr->strLen] = ch;
	strPtr->str[strPtr->strLen + 1] = '\0';
	strPtr->strLen++;

RET:
	LOG("exit");
	return ret;
}

