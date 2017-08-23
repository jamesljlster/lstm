#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_str.h"

#include "debug.h"

void lstm_str_trim(struct LSTM_STR* strPtr, const char* trimChs)
{
	int i, j;
	int stat, chStat;
	int validIndex = 0;
	int trimLen;
	int strLen;

	char* tmpPtr;

	LOG("enter");

	// Checking
	if(strPtr->strLen == 0)
	{
		return;
	}

	// Get reference
	tmpPtr = strPtr->str;

	// Get string length
	trimLen = strlen(trimChs);
	strLen = strlen(strPtr->str);

	// Find first valid character
	stat = 1;
	for(i = 0; i < strLen; i++)
	{
		chStat = 0;
		for(j = 0; j < trimLen; j++)
		{
			if(tmpPtr[i] == trimChs[j])
			{
				chStat = 1;
				break;
			}
		}

		if(chStat == 0)
		{
			stat = 0;
			validIndex = i;
			break;
		}
	}

	// Checking status
	if(stat == 1)
	{
		tmpPtr[0] = '\0';
		strPtr->strLen = 0;
		return;
	}

	// Move string
	for(i = 0, j = validIndex; j < strLen; i++, j++)
	{
		tmpPtr[i] = tmpPtr[j];
	}

	// Find last valid character
	stat = 1;
	for(i = strLen - 1; i >= 0; i++)
	{
		chStat = 0;
		for(j = 0; j < trimLen; j++)
		{
			if(tmpPtr[i] == trimChs[j])
			{
				chStat = 1;
				break;
			}
		}

		if(chStat == 0)
		{
			stat = 0;
			validIndex = i;
			break;
		}
	}

	// Checking status
	if(stat == 1)
	{
		tmpPtr[0] = '\0';
		strPtr->strLen = 0;
		return;
	}

	// Set terminate character
	tmpPtr[validIndex + 1] = '\0';

	// Find new string length
	strPtr->strLen = strlen(strPtr->str);

	LOG("exit");
}

int lstm_strcmp(const char* src1, const char* src2)
{
	int cmpLen;
	int srcLen;
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	cmpLen = strlen(src1);
	srcLen = strlen(src2);
	if(cmpLen != srcLen)
	{
		ret = LSTM_PARSE_FAILED;
	}
	else
	{
		ret = strncmp(src1, src2, cmpLen);
	}

	LOG("exit");
	return ret;
}

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

