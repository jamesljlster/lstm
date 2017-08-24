#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"
#include "lstm_str.h"

#include "debug.h"

#define LSTM_XML_TRIM_STR	" \t\r\n"

int lstm_xml_parse(struct LSTM_XML* xmlPtr, const char* filePath)
{
	int i;
	int ret = LSTM_NO_ERROR;
	int procIndex;

	int xmlLen;
	char* xml = NULL;
	char** strList = NULL;
	char** tmpPtr = NULL;

	struct LSTM_XML tmpXml;

	LOG("enter");

	// Zero memory
	memset(&tmpXml, 0, sizeof(struct LSTM_XML));

	// Read file to end
	ret = lstm_xml_fread_to_end(&xml, &xmlLen, filePath);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Create xml string list
	ret = lstm_xml_get_strlist(&strList, xml, xmlLen);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}
	else
	{
		tmpPtr = strList;
	}

	// Parse header
	ret = lstm_xml_parse_header(&tmpXml, (const char**)tmpPtr, &tmpPtr);
	if(ret != LSTM_NO_ERROR)
	{
		goto ERR;
	}

	// Assign value
	*xmlPtr = tmpXml;

	goto RET;

ERR:
	lstm_xml_delete(&tmpXml);

RET:
	lstm_free(xml);

	LOG("exit");
	return ret;
}

int lstm_xml_parse_element(struct LSTM_XML_ELEM** elemListPtr, int* elemLenPtr, const char* xmlSrc, int xmlLen, int* procIndex)
{
	int i;
	int tmpIndex = 0;
	int ret = LSTM_NO_ERROR;
	int tmpStat;

	char** strList = NULL;
	int strCount = 0;

	struct LSTM_XML_ELEM tmpElem;

	int childElemLen = 0;
	struct LSTM_XML_ELEM* childElemList = NULL;

	int tmpAttrLen = 0;
	struct LSTM_XML_ATTR* tmpAttrList = NULL;

	struct LSTM_STR strBuf;

	LOG("enter");

	// Zero memory
	memset(&strBuf, 0, sizeof(struct LSTM_STR));
	memset(&tmpElem, 0, sizeof(struct LSTM_XML_ELEM));

	// Set start processing index
	tmpIndex = *procIndex;

	// Find "<"
	tmpStat = 0;
	for(i = tmpIndex; i < xmlLen && tmpStat == 0; i++)
	{
		switch(xmlSrc[i])
		{
			case '<':
				tmpIndex = i + 1;
				tmpStat = 1;
				break;

			case '\t':
			case ' ':
				break;

			default:
				ret = LSTM_PARSE_FAILED;
				goto RET;
		}
	}

	// Checking
	if(tmpStat == 0)
	{
		goto RET;
	}

	// Read to ">"
	tmpStat = 0;
	for(i = tmpIndex; i < xmlLen && tmpStat == 0; i++)
	{
		switch(xmlSrc[i])
		{
			case '>':
				tmpIndex = i + 1;
				tmpStat = 1;
				break;

			default:
				ret = lstm_str_append(&strBuf, xmlSrc[i]);
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

	// Set procIndex
	*procIndex = tmpIndex;

	// Split string
	ret = lstm_xml_split(&strList, &strCount, strBuf.str);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Find attribute count
	if((strCount - 1) % 3 != 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}
	else
	{
		tmpAttrLen = (strCount - 1) / 3;
	}

	// Memory allocation: attribute list
	lstm_alloc(tmpAttrList, tmpAttrLen, struct LSTM_XML_ATTR, ret, ERR);
	tmpIndex = 1;
	for(i = 0; i < tmpAttrLen; i++)
	{
		// Checking
		ret = lstm_strcmp(strList[tmpIndex + 1], "=");
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Set value
		tmpAttrList[i].name = strList[tmpIndex];
		tmpAttrList[i].content = strList[tmpIndex + 2];

		strList[tmpIndex] = NULL;
		strList[tmpIndex + 2] = NULL;

		tmpIndex += 3;
	}

	// Assign tag
	tmpElem.name = strList[0];
	strList[0] = NULL;

	// Assign attribute
	tmpElem.attrList = tmpAttrList;
	tmpElem.attrLen = tmpAttrLen;
	tmpAttrList = NULL;
	tmpAttrLen = 0;

	goto RET;

ERR:
	lstm_xml_elem_delete(&tmpElem);

RET:
	// Delete attribute list
	for(i = 0; i < tmpAttrLen; i++)
	{
		lstm_xml_attr_delete(&tmpAttrList[i]);
	}
	lstm_free(tmpAttrList);

	// Delete child element list
	for(i = 0; i < childElemLen; i++)
	{
		lstm_xml_elem_delete(&childElemList[i]);
	}
	lstm_free(childElemList);

	// Delete string list
	for(i = 0; i < strCount; i++)
	{
		lstm_free(strList[i]);
	}
	lstm_free(strList);
	lstm_free(strBuf.str);

	LOG("exit");
	return ret;
}

int lstm_xml_parse_attribute(char** tagPtr, struct LSTM_XML_ATTR** attrListPtr, int* attrLenPtr, const char* attrStr)
{
	int i, tmpIndex;
	int ret = LSTM_NO_ERROR;

	int strCount = 0;
	char** strList = NULL;

	int tmpAttrLen = 0;
	struct LSTM_XML_ATTR* tmpAttrList = NULL;

	LOG("enter");

	// Split string
	ret = lstm_xml_split(&strList, &strCount, attrStr);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Find attribute count
	if((strCount - 1) % 3 != 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}
	else
	{
		tmpAttrLen = (strCount - 1) / 3;
	}

	// Memory allocation: attribute list
	lstm_alloc(tmpAttrList, tmpAttrLen, struct LSTM_XML_ATTR, ret, ERR);
	tmpIndex = 1;
	for(i = 0; i < tmpAttrLen; i++)
	{
		// Checking
		ret = lstm_strcmp(strList[tmpIndex + 1], "=");
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Set value
		tmpAttrList[i].name = strList[tmpIndex];
		tmpAttrList[i].content = strList[tmpIndex + 2];

		strList[tmpIndex] = NULL;
		strList[tmpIndex + 2] = NULL;

		tmpIndex += 3;
	}

	// Assign value
	*attrListPtr = tmpAttrList;
	*attrLenPtr = tmpAttrLen;
	*tagPtr = strList[0];
	strList[0] = NULL;

	goto RET;

ERR:
	for(i = 0; i < tmpAttrLen; i++)
	{
		lstm_xml_attr_delete(&tmpAttrList[i]);
	}
	lstm_free(tmpAttrList);

RET:
	for(i = 0; i < strCount; i++)
	{
		lstm_free(strList[i]);
	}
	lstm_free(strList);

	LOG("exit");
	return ret;
}

int lstm_xml_parse_header(struct LSTM_XML* xmlPtr, const char** strList, char*** endPtr)
{
	int i, tmpIndex;
	int ret = LSTM_NO_ERROR;

	char* tagPtr = NULL;
	int tmpAttrLen = 0;
	struct LSTM_XML_ATTR* tmpAttrList = NULL;

	char** xmlStrList;

	char** tmpStrList = NULL;
	int tmpStrCount = 0;

	struct LSTM_STR tmpStr;

	LOG("enter");

	// Zero memory
	memset(&tmpStr, 0, sizeof(struct LSTM_STR));

	// Cast pointer
	xmlStrList = (char**)strList;

	// Checking
	if(xmlStrList[0] == NULL)
	{
		goto RET;
	}

	// Compare header
	if(xmlStrList[0][0] == '<')
	{
		if(xmlStrList[0][1] != '?')
		{
			goto RET;
		}
	}
	else
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}

	// Processing header
	ret = lstm_str_create(&tmpStr, xmlStrList[0]);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}
	else
	{
		lstm_str_trim(&tmpStr, "<>?");
	}

	// Parsing attribute
	ret = lstm_xml_parse_attribute(&tagPtr, &tmpAttrList, &tmpAttrLen, tmpStr.str);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Check header tag
	ret = lstm_strcmp(tagPtr, "xml");
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Assign value
	xmlPtr->header = tmpAttrList;
	xmlPtr->headLen = tmpAttrLen;
	*endPtr = &xmlStrList[1];

	/*

	// Split string
	ret = lstm_xml_split(&tmpStrList, &tmpStrCount, tmpStr.str);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}


	// Find attribute count
	if((tmpStrCount - 1) % 3 != 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}
	else
	{
		tmpAttrLen = (tmpStrCount - 1) / 3;
	}

	// Memory allocation: attribute list
	lstm_alloc(tmpAttrList, tmpAttrLen, struct LSTM_XML_ATTR, ret, ERR);
	tmpIndex = 1;
	for(i = 0; i < tmpAttrLen; i++)
	{
		// Checking
		ret = lstm_strcmp(tmpStrList[tmpIndex + 1], "=");
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Set value
		tmpAttrList[i].name = tmpStrList[tmpIndex];
		tmpAttrList[i].content = tmpStrList[tmpIndex + 2];

		tmpStrList[tmpIndex] = NULL;
		tmpStrList[tmpIndex + 2] = NULL;

		tmpIndex += 3;
	}

	*/

	goto RET;

ERR:
	for(i = 0; i < tmpAttrLen; i++)
	{
		lstm_xml_attr_delete(&tmpAttrList[i]);
	}
	lstm_free(tmpAttrList);

RET:
	for(i = 0; i < tmpStrCount; i++)
	{
		lstm_free(tmpStrList[i]);
	}
	lstm_free(tmpStrList);

	lstm_free(tmpStr.str);
	lstm_free(tagPtr);

	LOG("exit");
	return ret;
}

/*
int lstm_xml_parse_header(struct LSTM_XML* xmlPtr, const char* xmlSrc, int xmlLen, int* procIndex)
{
	int i;
	int tmpIndex = 0;
	int ret = LSTM_NO_ERROR;
	int tmpStat;

	char** strList = NULL;
	int strCount = 0;

	int tmpAttrLen = 0;
	struct LSTM_XML_ATTR* tmpAttrList = NULL;

	struct LSTM_STR strBuf;

	LOG("enter");

	// Zero memory
	memset(&strBuf, 0, sizeof(struct LSTM_STR));

	// Set start processing index
	tmpIndex = *procIndex;

	// Find "<?"
	tmpStat = 0;
	for(i = tmpIndex; i < xmlLen && tmpStat == 0; i++)
	{
		switch(xmlSrc[i])
		{
			case '<':
				if(i < xmlLen - 1)
				{
					if(xmlSrc[i + 1] == '?')
					{
						tmpIndex = i + 2;
						tmpStat = 1;
					}
					else
					{
						ret = LSTM_PARSE_FAILED;
						goto RET;
					}
				}
				break;

			case '\t':
			case ' ':
				break;

			default:
				ret = LSTM_PARSE_FAILED;
				goto RET;
		}
	}

	// Checking
	if(tmpStat == 0)
	{
		goto RET;
	}

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
						tmpIndex = i + 2;
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
				ret = lstm_str_append(&strBuf, xmlSrc[i]);
				if(ret != LSTM_NO_ERROR)
				{
					goto RET;
				}
		}
	}

	// Set procIndex
	*procIndex = tmpIndex;

	// Checking
	if(tmpStat == 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}

	// Split string
	ret = lstm_xml_split(&strList, &strCount, strBuf.str);
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Check header tag
	ret = lstm_strcmp(strList[0], "xml");
	if(ret != LSTM_NO_ERROR)
	{
		goto RET;
	}

	// Find attribute count
	if((strCount - 1) % 3 != 0)
	{
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}
	else
	{
		tmpAttrLen = (strCount - 1) / 3;
	}

	// Memory allocation: attribute list
	lstm_alloc(tmpAttrList, tmpAttrLen, struct LSTM_XML_ATTR, ret, ERR);
	tmpIndex = 1;
	for(i = 0; i < tmpAttrLen; i++)
	{
		// Checking
		ret = lstm_strcmp(strList[tmpIndex + 1], "=");
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Set value
		tmpAttrList[i].name = strList[tmpIndex];
		tmpAttrList[i].content = strList[tmpIndex + 2];

		strList[tmpIndex] = NULL;
		strList[tmpIndex + 2] = NULL;

		tmpIndex += 3;
	}

	// Assign value
	xmlPtr->header = tmpAttrList;
	xmlPtr->headLen = tmpAttrLen;

	goto RET;

ERR:
	for(i = 0; i < tmpAttrLen; i++)
	{
		lstm_xml_attr_delete(&tmpAttrList[i]);
	}
	lstm_free(tmpAttrList);

RET:
	for(i = 0; i < strCount; i++)
	{
		lstm_free(strList[i]);
	}
	lstm_free(strList);
	lstm_free(strBuf.str);

	LOG("exit");
	return ret;
}
*/

#define __lstm_xml_strlist_append() \
	if(strBuf.strLen > 0) \
	{ \
		allocTmp = realloc(strList, sizeof(char*) * (strCount + 1)); \
		if(allocTmp == NULL) \
		{ \
			ret = LSTM_MEM_FAILED; \
			goto ERR; \
		} \
		else \
		{ \
			strList = allocTmp; \
			strCount++; \
		} \
		strList[strCount - 1] = strBuf.str; \
		memset(&strBuf, 0, sizeof(struct LSTM_STR)); \
	}

int lstm_xml_get_strlist(char*** strListPtr, const char* xmlSrc, int xmlLen)
{
	int i;
	int ret = LSTM_NO_ERROR;

	int strCount = 0;
	char** strList = NULL;

	struct LSTM_XML_PSTAT pStat;

	struct LSTM_STR strBuf;
	void* allocTmp;

	LOG("enter");

	// Zero memory
	memset(&pStat, 0, sizeof(struct LSTM_XML_PSTAT));
	memset(&strBuf, 0, sizeof(struct LSTM_STR));

#define __lstm_xml_assert(cond, retVal, errLabel) \
	if(cond) \
	{ \
		retVal = LSTM_PARSE_FAILED; \
		goto errLabel; \
	}

	// Split string
	for(i = 0; i < xmlLen; i++)
	{
		switch(xmlSrc[i])
		{
			case '<':
				__lstm_xml_assert(pStat.brStr, ret, ERR);
				pStat.brStr = 1;
				pStat.aStrB = 1;
				break;

			case '>':
				__lstm_xml_assert(!pStat.brStr, ret, ERR);
				pStat.brStr = 0;
				pStat.aStrA = 1;
				break;

			case ' ':
			case '\t':
				break;
		}

		// Append string to list before append character
		if(pStat.aStrB)
		{
			lstm_str_trim(&strBuf, LSTM_XML_TRIM_STR);
			__lstm_xml_strlist_append();
			pStat.aStrB = 0;
		}

		// Append character to string
		ret = lstm_str_append(&strBuf, xmlSrc[i]);
		if(ret != LSTM_NO_ERROR)
		{
			goto ERR;
		}

		// Append string to list after append character
		if(pStat.aStrA)
		{
			lstm_str_trim(&strBuf, LSTM_XML_TRIM_STR);
			__lstm_xml_strlist_append();
			pStat.aStrA = 0;
		}
	}

	// Append NULL to list
	allocTmp = realloc(strList, sizeof(char*) * (strCount + 1));
	if(allocTmp == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto ERR;
	}
	else
	{
		strList = allocTmp;
		strList[strCount] = NULL;
	}

	// Assign value
	*strListPtr = strList;

	goto RET;

ERR:
	if(strList != NULL)
	{
		for(i = 0; i < strCount; i++)
		{
			lstm_free(strList[i]);
		}
		lstm_free(strList);
	}

RET:
	lstm_free(strBuf.str);
	LOG("exit");

	return ret;
}

int lstm_xml_split(char*** strListPtr, int* strCountPtr, const char* src)
{
	int ret = LSTM_NO_ERROR;
	int finish;
	int procIndex;
	int forceRead;

	int strCount = 0;
	char** strList = NULL;

	struct LSTM_STR strBuf;
	void* allocTmp;

	LOG("enter");

	// Zero memory
	memset(&strBuf, 0, sizeof(struct LSTM_STR));

	// Split string
	forceRead = 0;
	procIndex = 0;
	finish = 0;
	while(finish == 0)
	{
		if(forceRead)
		{
			if(src[procIndex] == '"')
			{
				forceRead = ~forceRead;
				__lstm_xml_strlist_append();
			}
			else
			{
				ret = lstm_str_append(&strBuf, src[procIndex]);
				if(ret != LSTM_NO_ERROR)
				{
					goto ERR;
				}
			}
		}
		else
		{
			if(src[procIndex] == '"')
			{
				forceRead = ~forceRead;
				__lstm_xml_strlist_append();
			}
			else if(src[procIndex] == '=')
			{
				__lstm_xml_strlist_append();

				ret = lstm_str_append(&strBuf, '=');
				if(ret != LSTM_NO_ERROR)
				{
					goto ERR;
				}

				__lstm_xml_strlist_append();
			}
			else if(src[procIndex] == ' ' || src[procIndex] == '\0')
			{
				__lstm_xml_strlist_append();
			}
			else
			{
				ret = lstm_str_append(&strBuf, src[procIndex]);
				if(ret != LSTM_NO_ERROR)
				{
					goto ERR;
				}
			}
		}

		// Check if end of string
		if(src[procIndex] == '\0')
		{
			finish = 1;
		}

		procIndex++;
	}

	// Assign value
	*strListPtr = strList;
	*strCountPtr = strCount;

	goto RET;

ERR:
	if(strList != NULL)
	{
		for(procIndex = 0; procIndex < strCount; procIndex++)
		{
			lstm_free(strList[procIndex]);
		}
		lstm_free(strList);
	}

RET:
	lstm_free(strBuf.str);
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

