#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"

#include "debug.h"

int lstm_xml_parse(struct LSTM_XML* xmlPtr, const char* filePath)
{
	int ret = LSTM_NO_ERROR;

	LOG("enter");

	goto RET;

RET:
	LOG("exit");
	return ret;
}

int lstm_xml_fread_to_end(char** strPtr, const char* filePath)
{
	int i, iResult;
	int ret = LSTM_NO_ERROR;
	int fileLen;
	char* tmpPtr = NULL;
	FILE* fRead = NULL;

	LOG("enter");

	// Open file
	fRead = fopen(filePath, "rb");
	if(fRead == NULL)
	{
		ret = LSTM_FILE_OP_FAILED;
		goto RET;
	}

	// Find file length
	fseek(fRead, 0, SEEK_END);
	fileLen = ftell(fRead);
	fseek(fRead, 0, SEEK_SET);

	// Memory allocation
	lstm_alloc(tmpPtr, fileLen + 1, char, ret, RET);

	// Read file
	for(i = 0; i < fileLen; i++)
	{
		iResult = fread(&tmpPtr[i], sizeof(char), 1, fRead);
		if(iResult != 1)
		{
			ret = LSTM_FILE_OP_FAILED;
			goto ERR;
		}
	}

	// Assign value
	*strPtr = tmpPtr;

	goto RET;

ERR:
	lstm_free(tmpPtr);

RET:
	if(fRead != NULL)
	{
		fclose(fRead);
	}

	LOG("exit"); 
	return ret;
}

