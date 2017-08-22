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

	int xmlLen;
	char* xml = NULL;

	struct LSTM_XML* tmpXmlPtr = NULL;

	LOG("enter");

	// Read file to end
	ret = lstm_xml_fread_to_end(&xml, &xmlLen, filePath);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Memory allocation: Root xml structure
	lstm_alloc(tmpXmlPtr, 1, struct LSTM_XML, ret, RET);

	goto RET;

ERR:
	if(tmpXmlPtr != NULL)
	{
		lstm_xml_delete(tmpXmlPtr);
		lstm_free(tmpXmlPtr);
	}

RET:
	lstm_free(xml);

	LOG("exit");
	return ret;
}

int lstm_xml_parse_header(struct LSTM_XML* xmlPtr, char* xmlSrc, int xmlLen, int* procIndex)
{
	int i;
	int tmpIndex = 0;
	int ret = LSTM_NO_ERROR;
	int tmpStat;

	char preChar = '\0';

	struct LSTM_XML_PSTAT pStat;
	struct LSTM_XML_STR strBuf;

	LOG("enter");

	// Zero memory
	memset(&pStat, 0, sizeof(struct LSTM_XML_PSTAT));
	memset(&strBuf, 0, sizeof(struct LSTM_XML_STR));

	// Find "<?"
	tmpStat = 0;
	for(i = 0; i < xmlLen && tmpStat == 0; i++)
	{
		switch(xmlSrc[i])
		{
			case '?':
				if(preChar == '<')
				{
					tmpIndex = i + 1;
					tmpStat = 1;
				}
				break;

			case '<':
			case '\t':
			case ' ':
				break;

			default:
				ret = LSTM_PARSE_FAILED;
				goto RET;
		}

		preChar = xmlSrc[i];
	}

	// Checking
	if(tmpStat == 0)
	{
		goto RET;
	}

	// Append character to string until "?>"
	tmpStat = 0;
	for(i = tmpIndex; i < xmlLen && tmpStat == 0; i++)
	{
		switch(xmlSrc[i])
		{
			case '?':
				if(i < xmlLen - 1)
				{
					if(xmlSrc[i + 1] == '>')
					{
						tmpIndex = i + 1;
						tmpStat = 1;
					}
					else
					{
						ret = LSTM_PARSE_FAILED;
						goto RET;
					}
				}
				break;

			default:
				ret = lstm_xml_str_append(&strBuf, xmlSrc[i]);
				if(ret != LSTM_NO_ERROR)
				{
					goto RET;
				}
		}
	}

	// Checking
	if(tmpStat == 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}

	printf("%s\n", strBuf.buf);

ERR:

RET:
	lstm_free(strBuf.buf);

	LOG("exit");
	return ret;
}

int lstm_xml_str_append(struct LSTM_XML_STR* strPtr, char ch)
{
	int ret = LSTM_MEM_FAILED;
	int tmpLen;

	void* allocTmp;

	LOG("enter");

	// Find new memory length
	tmpLen = strPtr->strLen + 2;

	// Reallocate memory
	if(tmpLen > strPtr->memLen)
	{
		allocTmp = realloc(strPtr->buf, tmpLen * sizeof(char));
		if(allocTmp == NULL)
		{
			ret = LSTM_MEM_FAILED;
			goto RET;
		}
	}

	// Append character
	strPtr->buf[strPtr->strLen] = ch;
	strPtr->strLen++;

RET:
	LOG("exit");
	return ret;
}

void lstm_xml_delete(struct LSTM_XML* xmlPtr)
{
	int i;

	LOG("enter");

	if(xmlPtr->header != NULL)
	{
		for(i = 0; i < xmlPtr->headLen; i++)
		{
			lstm_xml_attr_delete(&xmlPtr->header[i]);
		}
		lstm_free(xmlPtr->header);
	}

	if(xmlPtr->elemList != NULL)
	{
		for(i = 0; i < xmlPtr->elemLen; i++)
		{
			lstm_xml_elem_delete(&xmlPtr->elemList[i]);
		}
		lstm_free(xmlPtr->elemList);
	}

	LOG("exit");
}

void lstm_xml_elem_delete(struct LSTM_XML_ELEM* xmlElemPtr)
{
	int i;

	LOG("enter");

	lstm_free(xmlElemPtr->name);
	lstm_free(xmlElemPtr->text);

	if(xmlElemPtr->attrList != NULL)
	{
		for(i = 0; i < xmlElemPtr->attrLen; i++)
		{
			lstm_xml_attr_delete(&xmlElemPtr->attrList[i]);
		}
		lstm_free(xmlElemPtr->attrList);
	}

	if(xmlElemPtr->elemList != NULL)
	{
		for(i = 0; i < xmlElemPtr->elemLen; i++)
		{
			lstm_xml_elem_delete(&xmlElemPtr->elemList[i]);
		}
		lstm_free(xmlElemPtr->elemList);
	}

	memset(xmlElemPtr, 0, sizeof(struct LSTM_XML_ELEM));

	LOG("exit");
}

void lstm_xml_attr_delete(struct LSTM_XML_ATTR* xmlAttrPtr)
{
	LOG("enter");

	lstm_free(xmlAttrPtr->name);
	lstm_free(xmlAttrPtr->content);

	memset(xmlAttrPtr, 0, sizeof(struct LSTM_XML_ATTR));

	LOG("exit");
}

int lstm_xml_fread_to_end(char** strPtr, int* lenPtr, const char* filePath)
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
	if(strPtr != NULL)
	{
		*strPtr = tmpPtr;
	}
	if(lenPtr != NULL)
	{
		*lenPtr = fileLen;
	}

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

