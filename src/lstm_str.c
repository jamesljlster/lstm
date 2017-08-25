#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_str.h"

#ifdef DEBUG
#undef DEBUG
#endif

#include "debug.h"

int lstm_str_create(struct LSTM_STR* strPtr, const char* src)
{
	int ret = LSTM_NO_ERROR;
	int strLen;
	struct LSTM_STR tmpStr;

	LOG("enter");

	// Zero memory
	memset(&tmpStr, 0, sizeof(struct LSTM_STR));

	// Memory allocation
	strLen = strlen(src);
	lstm_alloc(tmpStr.str, strLen + 1, char, ret, RET);

	// Copy string
	strncpy(tmpStr.str, src, strLen);

	// Assign value
	tmpStr.strLen = strLen;
	tmpStr.memLen = strLen + 1;
	*strPtr = tmpStr;

RET:
	LOG("exit");
	return ret;
}

int lstm_str_index_of(const char* src, char ch)
{
	int i;
	int ret = LSTM_PARSE_FAILED;
	int strLen;

	LOG("enter");

	// Find string length
	strLen = strlen(src);

	for(i = 0; i < strLen; i++)
	{
		if(src[i] == ch)
		{
			ret = i;
			break;
		}
	}

	LOG("exit");

	return ret;
}

void lstm_str_trim(struct LSTM_STR* strPtr, const char* trimChs)
{
	int i, j;
	int stat;
	int validIndex = 0;
	int strLen;

	char* tmpPtr;

	LOG("enter");

	// Checking
	if(strPtr->strLen == 0)
	{
		goto RET;
	}

	// Get reference
	tmpPtr = strPtr->str;

	// Get string length
	strLen = strlen(strPtr->str);

	// Find first valid character
	stat = 1;
	for(i = 0; i < strLen; i++)
	{
		if(lstm_str_index_of(trimChs, tmpPtr[i]) < 0)
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
		goto RET;
	}

	// Move string
	for(i = 0, j = validIndex; j < strLen + 1; i++, j++)
	{
		tmpPtr[i] = tmpPtr[j];
	}

	// Find last valid character
	strLen = strlen(tmpPtr);
	stat = 1;
	for(i = strLen - 1; i >= 0; i--)
	{
		if(lstm_str_index_of(trimChs, tmpPtr[i]) < 0)
		{
			stat = 0;
			validIndex = i;
			break;
		}
	}

	// Checking status
	if(stat == 0)
	{
		tmpPtr[validIndex + 1] = '\0';
	}

	// Find new string length
	strPtr->strLen = strlen(strPtr->str);

RET:
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
			strPtr->memLen = tmpLen;
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

