#ifndef __LSTM_STRDEF_H__
#define __LSTM_STRDEF_H__

// LSTM reserve words enumeration
enum LSTM_STR_LIST
{
	LSTM_STR_CONFIG,
	LSTM_STR_FORGET_GATE,
	LSTM_STR_HIDDEN,
	LSTM_STR_H_LAYER,
	LSTM_STR_INDEX,
	LSTM_STR_INPUT,
	LSTM_STR_INPUTS,
	LSTM_STR_INPUT_GATE,
	LSTM_STR_LAYER,
	LSTM_STR_LAYERS,
	LSTM_STR_LRATE,
	LSTM_STR_MCOEF,
	LSTM_STR_MODEL,
	LSTM_STR_NETWORK,
	LSTM_STR_NODE,
	LSTM_STR_NODES,
	LSTM_STR_OUTPUT,
	LSTM_STR_OUTPUTS,
	LSTM_STR_OUTPUT_GATE,
	LSTM_STR_RECURRENT,
	LSTM_STR_TFUNC,
	LSTM_STR_THRESHOLD,
	LSTM_STR_TYPE,
	LSTM_STR_VALUE,
	LSTM_STR_WEIGHT,

	LSTM_STR_AMOUNT
};

// LSTM reserve words list
extern const char* lstm_str_list[];

#ifdef __cplusplus
extern "C" {
#endif

int lstm_strdef_get_id(const char* str);

#ifdef __cplusplus
}
#endif

#endif
