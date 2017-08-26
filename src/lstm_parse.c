#include <stdlib.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"
#include "lstm_strdef.h"

#include "debug.h"

#define __lstm_strtol(num, src, retVal, errLabel) \
	num = strtol(src, &tmpPtr, 10); \
	if(src == tmpPtr) \
	{ \
		retVal = LSTM_PARSE_FAILED; \
		goto errLabel; \
	}

#define __lstm_strtod(num, src, retVal, errLabel) \
	num = strtod(src, &tmpPtr); \
	if(src == tmpPtr) \
	{ \
		retVal = LSTM_PARSE_FAILED; \
		goto errLabel; \
	}

int lstm_parse_config_tfun_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	int tmpVal;
	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;

	// Parsing
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_INPUT:
				break;

			case LSTM_STR_OUTPUT:
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_config_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	double tmpdVal;
	int tmpVal;
	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;

	// Parsing
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_INPUTS:
				__lstm_strtol(tmpVal, childElemPtr[i].text, ret, RET);
				lstm_run(lstm_config_set_inputs(lstmCfg, tmpVal), ret, RET);
				break;

			case LSTM_STR_OUTPUTS:
				__lstm_strtol(tmpVal, childElemPtr[i].text, ret, RET);
				lstm_run(lstm_config_set_outputs(lstmCfg, tmpVal), ret, RET);
				break;

			case LSTM_STR_TFUNC:
				break;

			case LSTM_STR_LRATE:
				__lstm_strtod(tmpdVal, childElemPtr[i].text, ret, RET);
				lstm_config_set_learning_rate(lstmCfg, tmpdVal);
				break;

			case LSTM_STR_MCOEF:
				__lstm_strtod(tmpdVal, childElemPtr[i].text, ret, RET);
				lstm_config_set_momentum_coef(lstmCfg, tmpdVal);
				break;

			case LSTM_STR_H_LAYER:
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}
