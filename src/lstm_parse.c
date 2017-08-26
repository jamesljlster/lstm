#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"
#include "lstm_strdef.h"
#include "lstm_parse.h"

#include "debug.h"

#define __lstm_strtol(num, src, retVal, errLabel) \
	num = strtol(src, &tmpPtr, 10); \
	if(src == tmpPtr) \
	{ \
		fprintf(stderr, "%s(): Failed to parse \"%s\" as integer!\n", __FUNCTION__, src); \
		retVal = LSTM_PARSE_FAILED; \
		goto errLabel; \
	}

#define __lstm_strtod(num, src, retVal, errLabel) \
	num = strtod(src, &tmpPtr); \
	if(src == tmpPtr) \
	{ \
		fprintf(stderr, "%s(): Failed to parse \"%s\" as real number!\n", __FUNCTION__, src); \
		retVal = LSTM_PARSE_FAILED; \
		goto errLabel; \
	}

int lstm_config_import(lstm_config_t* lstmCfgPtr, const char* filePath)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_XML xml;
	struct LSTM_XML_ELEM* elemPtr = NULL;

	lstm_config_t tmpCfg = NULL;

	// Zero memory
	memset(&xml, 0, sizeof(struct LSTM_XML));

	// Parse xml
	lstm_run(lstm_xml_parse(&xml, filePath), ret, RET);

	// Create config
	lstm_run(lstm_config_create(&tmpCfg), ret, RET);

	// Find xml config node
	elemPtr = lstm_xml_get_element_root(&xml, lstm_str_list[LSTM_STR_MODEL]);
	if(elemPtr == NULL)
	{
		LOG("%s not found in xml!", lstm_str_list[LSTM_STR_MODEL]);
		ret = LSTM_PARSE_FAILED;
		goto ERR;
	}

	elemPtr = lstm_xml_get_element(elemPtr, lstm_str_list[LSTM_STR_CONFIG]);
	if(elemPtr == NULL)
	{
		LOG("%s not found in xml!", lstm_str_list[LSTM_STR_CONFIG]);
		ret = LSTM_PARSE_FAILED;
		goto ERR;
	}

	// Parse config
	lstm_run(lstm_parse_config_xml(tmpCfg, elemPtr), ret, ERR);

	// Assign value
	*lstmCfgPtr = tmpCfg;
	goto RET;

ERR:
	lstm_config_delete(tmpCfg);

RET:
	lstm_xml_delete(&xml);

	LOG("exit");
	return ret;
}

int lstm_parse_config_hidden_nodes_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	int layerIndex;
	int nodes;
	char* tmpPtr;

	LOG("enter");

	// Parse attribute
	for(i = 0; i < elemPtr->attrLen; i++)
	{
		strId = lstm_strdef_get_id(elemPtr->attrList[i].name);
		switch(strId)
		{
			case LSTM_STR_INDEX:
				// Parse layer index
				__lstm_strtol(layerIndex, elemPtr->attrList[i].content, ret, RET);

				// Parse nodes
				__lstm_strtol(nodes, elemPtr->text, ret, RET);

				// Set hidden nodes
				lstm_run(lstm_config_set_hidden_nodes(lstmCfg, layerIndex, nodes), ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_config_hidden_layer_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr)
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

	// Parse attribute
	for(i = 0; i < elemPtr->attrLen; i++)
	{
		strId = lstm_strdef_get_id(elemPtr->attrList[i].name);
		switch(strId)
		{
			case LSTM_STR_LAYERS:
				__lstm_strtol(tmpVal, elemPtr->attrList[i].content, ret, RET);
				lstm_run(lstm_config_set_hidden_layers(lstmCfg, tmpVal), ret, RET);
				break;
		}
	}

	// Parse element
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_NODES:
				lstm_run(lstm_parse_config_hidden_nodes_xml(lstmCfg, &childElemPtr[i]), ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_config_tfun_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	int tmpVal;

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
				tmpVal = lstm_get_transfer_func_id(childElemPtr[i].text);
				lstm_run(lstm_config_set_input_transfer_func(lstmCfg, tmpVal), ret, RET);
				break;

			case LSTM_STR_OUTPUT:
				tmpVal = lstm_get_transfer_func_id(childElemPtr[i].text);
				lstm_run(lstm_config_set_output_transfer_func(lstmCfg, tmpVal), ret, RET);
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
				lstm_run(lstm_parse_config_tfun_xml(lstmCfg, &childElemPtr[i]), ret, RET);
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
				lstm_run(lstm_parse_config_hidden_layer_xml(lstmCfg, &childElemPtr[i]), ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

