#include <stdlib.h>
#include <assert.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_builtin_math.h"
#include "lstm_xml.h"
#include "lstm_strdef.h"
#include "lstm_print.h"

#include "debug.h"

#define __lstm_fprint_indent(indent, fstr, ...) \
	lstm_xml_fprint_indent(fptr, indent); \
	fprintf(fptr, fstr, ##__VA_ARGS__)

void lstm_fprint_vector(FILE* fptr, double* vector, int vectorLen, int indent)
{
	int i;

	for(i = 0; i < vectorLen; i++)
	{
		__lstm_fprint_indent(indent, "<%s %s=\"%d\"> %.32lf </%s>\n",
				lstm_str_list[LSTM_STR_VALUE],
				lstm_str_list[LSTM_STR_INDEX], i,
				vector[i],
				lstm_str_list[LSTM_STR_VALUE]);
	}
}

void lstm_fprint_base(FILE* fptr, struct LSTM_BASE* basePtr, int netLen, int layerIndex, int indent)
{
	// Print weight
	__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_WEIGHT]);
	lstm_fprint_vector(fptr, basePtr->weight, netLen, indent + 1);
	__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_WEIGHT]);

	// Print recurrent weight
	if(layerIndex == 1)
	{
		__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_RECURRENT]);
		lstm_fprint_vector(fptr, basePtr->rWeight, netLen, indent + 1);
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_RECURRENT]);
	}

	// Print threshold
	__lstm_fprint_indent(indent, "<%s> %.32lf </%s>\n",
			lstm_str_list[LSTM_STR_THRESHOLD],
			basePtr->th,
			lstm_str_list[LSTM_STR_THRESHOLD]);
}

void lstm_fprint_node(FILE* fptr, struct LSTM_STRUCT* lstm, int layerIndex, int nodeIndex, int indent)
{
	int netLen;
	struct LSTM_NODE* nodeRef;

	// Get reference
	nodeRef = &lstm->layerList[layerIndex].nodeList[nodeIndex];
	netLen = lstm->layerList[layerIndex - 1].nodeCount;

	// Print base networks
	if(layerIndex < lstm->config.layers - 1)
	{
		// Input network
		__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_INPUT]);
		lstm_fprint_base(fptr, &nodeRef->inputNet, netLen, layerIndex, indent + 1);
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_INPUT]);

		// Input gate
		__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_INPUT_GATE]);
		lstm_fprint_base(fptr, &nodeRef->igNet, netLen, layerIndex, indent + 1);
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_INPUT_GATE]);

		// Forget gate
		__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_FORGET_GATE]);
		lstm_fprint_base(fptr, &nodeRef->fgNet, netLen, layerIndex, indent + 1);
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_FORGET_GATE]);

		// Output gate
		__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_OUTPUT_GATE]);
		lstm_fprint_base(fptr, &nodeRef->ogNet, netLen, layerIndex, indent + 1);
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_OUTPUT_GATE]);
	}
	else
	{
		lstm_fprint_base(fptr, &nodeRef->inputNet, netLen, layerIndex, indent);
	}
}

void lstm_fprint_layer(FILE* fptr, struct LSTM_STRUCT* lstm, int layerIndex, int indent)
{
	int i;

	// Print nodes
	for(i = 0; i < lstm->layerList[layerIndex].nodeCount; i++)
	{
		// Print node
		__lstm_fprint_indent(indent, "<%s %s=\"%d\">\n",
				lstm_str_list[LSTM_STR_NODE],
				lstm_str_list[LSTM_STR_INDEX], i);

		lstm_fprint_node(fptr, lstm, layerIndex, i, indent + 1);

		// Print end node
		__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_NODE]);
	}

}

void lstm_fprint_net(FILE* fptr, struct LSTM_STRUCT* lstm, int indent)
{
	int i;

	// Print network
	__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_NETWORK]);

	// Print layers
	for(i = 1; i < lstm->config.layers; i++)
	{
		// Print layer
		if(i < lstm->config.layers - 1)
		{
			__lstm_fprint_indent(indent + 1, "<%s %s=\"%s\" %s=\"%d\">\n",
					lstm_str_list[LSTM_STR_LAYER],
					lstm_str_list[LSTM_STR_TYPE], lstm_str_list[LSTM_STR_HIDDEN],
					lstm_str_list[LSTM_STR_INDEX], i - 1);

			// Print nodes
			lstm_fprint_layer(fptr, lstm, i, indent + 2);

			__lstm_fprint_indent(indent + 1, "</%s>\n", lstm_str_list[LSTM_STR_LAYER]);
		}
		else
		{
			__lstm_fprint_indent(indent + 1, "<%s %s=\"%s\">\n",
					lstm_str_list[LSTM_STR_LAYER],
					lstm_str_list[LSTM_STR_TYPE], lstm_str_list[LSTM_STR_OUTPUT]);

			// Print nodes
			lstm_fprint_layer(fptr, lstm, i, indent + 2);

			__lstm_fprint_indent(indent + 1, "</%s>\n", lstm_str_list[LSTM_STR_LAYER]);
		}
	}

	// Print end network
	__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_NETWORK]);
}

void lstm_fprint_config(FILE* fptr, struct LSTM_CONFIG_STRUCT* lstmCfg, int indent)
{
	int i;
	const char* tmpPtr;

	// Print config
	__lstm_fprint_indent(indent, "<%s>\n", lstm_str_list[LSTM_STR_CONFIG]);

	// Print inputs, outputs
	tmpPtr = lstm_str_list[LSTM_STR_INPUTS];
	__lstm_fprint_indent(indent + 1, "<%s> %d </%s>\n", tmpPtr, lstmCfg->inputs, tmpPtr);

	tmpPtr = lstm_str_list[LSTM_STR_OUTPUTS];
	__lstm_fprint_indent(indent + 1, "<%s> %d </%s>\n", tmpPtr, lstmCfg->outputs, tmpPtr);

	// Print transfer function
	__lstm_fprint_indent(indent + 1, "<%s>\n", lstm_str_list[LSTM_STR_TFUNC]);

	tmpPtr = lstm_str_list[LSTM_STR_INPUT];
	__lstm_fprint_indent(indent + 2, "<%s> %s </%s>\n", tmpPtr,
			lstm_transfer_func_name[lstmCfg->inputTFunc], tmpPtr);

	tmpPtr = lstm_str_list[LSTM_STR_OUTPUT];
	__lstm_fprint_indent(indent + 2, "<%s> %s </%s>\n", tmpPtr,
			lstm_transfer_func_name[lstmCfg->outputTFunc], tmpPtr);

	__lstm_fprint_indent(indent + 1, "</%s>\n", lstm_str_list[LSTM_STR_TFUNC]);

	// Print learning rate and momentum coefficient
	tmpPtr = lstm_str_list[LSTM_STR_LRATE];
	__lstm_fprint_indent(indent + 1, "<%s> %lf </%s>\n", tmpPtr, lstmCfg->lRate, tmpPtr);

	tmpPtr = lstm_str_list[LSTM_STR_MCOEF];
	__lstm_fprint_indent(indent + 1, "<%s> %lf </%s>\n", tmpPtr, lstmCfg->mCoef, tmpPtr);

	// Print hidden layer
	__lstm_fprint_indent(indent + 1, "<%s %s=\"%d\">\n",
			lstm_str_list[LSTM_STR_H_LAYER],
			lstm_str_list[LSTM_STR_LAYERS],
			lstmCfg->layers - 2);

	// Print hidden nodes
	for(i = 0; i < lstmCfg->layers - 2; i++)
	{
		tmpPtr = lstm_str_list[LSTM_STR_VALUE];
		__lstm_fprint_indent(indent + 2, "<%s %s=\"%d\"> %d </%s>\n",
				tmpPtr, lstm_str_list[LSTM_STR_INDEX], i, lstmCfg->nodeList[i + 1], tmpPtr);
	}

	// Print end hidden layer
	__lstm_fprint_indent(indent + 1, "</%s>\n", lstm_str_list[LSTM_STR_H_LAYER]);

	// Print end config
	__lstm_fprint_indent(indent, "</%s>\n", lstm_str_list[LSTM_STR_CONFIG]);
}

