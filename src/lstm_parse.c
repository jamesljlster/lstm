#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"
#include "lstm_strdef.h"
#include "lstm_parse.h"

#include "debug.h"

#ifdef DEBUG
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
#else
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
#endif

int lstm_import(lstm_t* lstmPtr, const char* filePath)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_XML xml;
	struct LSTM_XML_ELEM* rootElemPtr = NULL;
	struct LSTM_XML_ELEM* elemPtr = NULL;

	lstm_config_t tmpCfg = NULL;
	lstm_t tmpLstm = NULL;

	// Zero memory
	memset(&xml, 0, sizeof(struct LSTM_XML));

	// Parse xml
	lstm_run(lstm_xml_parse(&xml, filePath), ret, RET);

	// Create config
	lstm_run(lstm_config_create(&tmpCfg), ret, RET);

	// Find lstm root element node
	rootElemPtr = lstm_xml_get_element_root(&xml, lstm_str_list[LSTM_STR_MODEL]);
	if(rootElemPtr == NULL)
	{
		LOG("%s not found in xml!", lstm_str_list[LSTM_STR_MODEL]);
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}

	// Find xml config node
	elemPtr = lstm_xml_get_element(rootElemPtr, lstm_str_list[LSTM_STR_CONFIG]);
	if(elemPtr == NULL)
	{
		LOG("%s not found in xml!", lstm_str_list[LSTM_STR_CONFIG]);
		ret = LSTM_PARSE_FAILED;
		goto RET;
	}

	// Parse config
	lstm_run(lstm_parse_config_xml(tmpCfg, elemPtr), ret, RET);

	// Create lstm networks
	lstm_run(lstm_create(&tmpLstm, tmpCfg), ret, ERR);

	// Find xml network node
	elemPtr = lstm_xml_get_element(rootElemPtr, lstm_str_list[LSTM_STR_NETWORK]);
	if(elemPtr == NULL)
	{
		LOG("%s not found in xml!", lstm_str_list[LSTM_STR_NETWORK]);
		ret = LSTM_PARSE_FAILED;
		goto ERR;
	}

	// Parse network
	lstm_run(lstm_parse_net_xml(tmpLstm, elemPtr), ret, ERR);

	// Assign value
	*lstmPtr = tmpLstm;
	goto RET;

ERR:
	lstm_delete(tmpLstm);

RET:
	lstm_config_delete(tmpCfg);
	lstm_xml_delete(&xml);

	LOG("exit");
	return ret;
}

int lstm_parse_net_vector_xml(double* vector, int vectorLen, struct LSTM_XML_ELEM* elemPtr)
{
	int i, j;
	int strId;
	int ret = LSTM_NO_ERROR;

	int index;
	double value;
	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;

	// Parsing element
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_VALUE:
				// Parsing attribute
				index = 0;
				for(j = 0; j < childElemPtr[i].attrLen; j++)
				{
					strId = lstm_strdef_get_id(childElemPtr[i].attrList[j].name);
					switch(strId)
					{
						case LSTM_STR_INDEX:
							__lstm_strtol(index, childElemPtr[i].attrList[j].content, ret, RET);
							break;
					}
				}

				// Checking
				if(index < 0 || index >= vectorLen)
				{
					ret = LSTM_OUT_OF_RANGE;
					goto RET;
				}

				// Set value
				__lstm_strtod(value, childElemPtr[i].text, ret, RET);
				vector[index] = value;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_net_base_xml(struct LSTM_BASE* basePtr, int netLen, int reNetLen, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;

	// Parsing element
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_WEIGHT:
				assert(basePtr->weight != NULL);
				lstm_run(lstm_parse_net_vector_xml(basePtr->weight, netLen, &childElemPtr[i]), ret, RET);
				break;

			case LSTM_STR_RECURRENT:
				if(reNetLen > 0)
				{
					assert(basePtr->rWeight != NULL);
					lstm_run(lstm_parse_net_vector_xml(basePtr->rWeight, reNetLen, &childElemPtr[i]), ret, RET);
				}
				break;

			case LSTM_STR_THRESHOLD:
				__lstm_strtod(basePtr->th, childElemPtr[i].text, ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_net_node_xml(struct LSTM_STRUCT* lstm, int layerIndex, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;
	int netLen, reNetLen;

	int nodeIndex = 0;

	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	struct LSTM_LAYER* layerRef;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;
	layerRef = lstm->layerList;

	assert(layerRef != NULL);

	// Parsing attribute
	for(i = 0; i < elemPtr->attrLen; i++)
	{
		strId = lstm_strdef_get_id(elemPtr->attrList[i].name);
		switch(strId)
		{
			case LSTM_STR_INDEX:
				__lstm_strtol(nodeIndex, elemPtr->attrList[i].content, ret, RET);
				break;
		}
	}

	// Checking
	if(nodeIndex >= lstm->layerList[layerIndex].nodeCount || nodeIndex < 0)
	{
		ret = LSTM_OUT_OF_RANGE;
		goto RET;
	}

	assert(layerRef[layerIndex].nodeList != NULL);

	// Find network length
	netLen = layerRef[layerIndex - 1].nodeCount;
	if(layerIndex == 1)
	{
		reNetLen = layerRef[lstm->config.layers - 2].nodeCount;
	}
	else
	{
		reNetLen = 0;
	}

	// Parse network base
	if(layerIndex < lstm->config.layers - 1)
	{
		// Parsing element
		for(i = 0; i < childElemLen; i++)
		{
			strId = lstm_strdef_get_id(childElemPtr[i].name);
			switch(strId)
			{
				case LSTM_STR_INPUT:
					lstm_run(lstm_parse_net_base_xml(
								&layerRef[layerIndex].nodeList[nodeIndex].inputNet,
								netLen, reNetLen,
								&childElemPtr[i]),
							ret, RET);
					break;

				case LSTM_STR_INPUT_GATE:
					lstm_run(lstm_parse_net_base_xml(
								&layerRef[layerIndex].nodeList[nodeIndex].igNet,
								netLen, reNetLen,
								&childElemPtr[i]),
							ret, RET);
					break;

				case LSTM_STR_FORGET_GATE:
					lstm_run(lstm_parse_net_base_xml(
								&layerRef[layerIndex].nodeList[nodeIndex].fgNet,
								netLen, reNetLen,
								&childElemPtr[i]),
							ret, RET);
					break;

				case LSTM_STR_OUTPUT_GATE:
					lstm_run(lstm_parse_net_base_xml(
								&layerRef[layerIndex].nodeList[nodeIndex].ogNet,
								netLen, reNetLen,
								&childElemPtr[i]),
							ret, RET);
					break;
			}
		}
	}
	else
	{
		lstm_run(lstm_parse_net_base_xml(
					&layerRef[layerIndex].nodeList[nodeIndex].inputNet,
					netLen, reNetLen,
					elemPtr),
				ret, RET);
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_net_layer_xml(struct LSTM_STRUCT* lstm, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

	int layerType = -1;
	int layerIndex = -1;

	char* tmpPtr;

	int childElemLen;
	struct LSTM_XML_ELEM* childElemPtr;

	LOG("enter");

	// Get reference
	childElemLen = elemPtr->elemLen;
	childElemPtr = elemPtr->elemList;

	// Parsing attribute
	for(i = 0; i < elemPtr->attrLen; i++)
	{
		strId = lstm_strdef_get_id(elemPtr->attrList[i].name);
		switch(strId)
		{
			case LSTM_STR_TYPE:
				layerType = lstm_strdef_get_id(elemPtr->attrList[i].content);
				break;

			case LSTM_STR_INDEX:
				__lstm_strtol(layerIndex, elemPtr->attrList[i].content, ret, RET);
				break;
		}
	}

	// Set layer index
	switch(layerType)
	{
		case LSTM_STR_HIDDEN:
			layerIndex = layerIndex + 1;
			break;

		case LSTM_STR_OUTPUT:
			layerIndex = lstm->config.layers - 1;
			break;

		default:
			ret = LSTM_INVALID_ARG;
			goto RET;
	}

	// Checking
	if(layerIndex >= lstm->config.layers || layerIndex < 1)
	{
		ret = LSTM_OUT_OF_RANGE;
		goto RET;
	}

	// Parsing element
	for(i = 0; i < childElemLen; i++)
	{
		strId = lstm_strdef_get_id(childElemPtr[i].name);
		switch(strId)
		{
			case LSTM_STR_NODE:
				lstm_run(lstm_parse_net_node_xml(lstm, layerIndex, &childElemPtr[i]), ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
}

int lstm_parse_net_xml(struct LSTM_STRUCT* lstm, struct LSTM_XML_ELEM* elemPtr)
{
	int i;
	int strId;
	int ret = LSTM_NO_ERROR;

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
			case LSTM_STR_LAYER:
				lstm_run(lstm_parse_net_layer_xml(lstm, &childElemPtr[i]), ret, RET);
				break;
		}
	}

RET:
	LOG("exit");
	return ret;
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

