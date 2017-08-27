
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

