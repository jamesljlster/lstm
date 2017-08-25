
#include "lstm.h"
#include "lstm_str.h"
#include "lstm_strdef.h"

#include "debug.h"

const char* lstm_str_list[] = {
	"config",
	"forget_gate",
	"hidden",
	"hidden_layer",
	"index",
	"input",
	"inputs",
	"input_gate",
	"layer",
	"layers",
	"learning_rate",
	"momentum_coefficient",
	"lstm_model",
	"network",
	"node",
	"nodes",
	"output",
	"outputs",
	"output_gate",
	"recurrent",
	"transfer_function",
	"threshold",
	"type",
	"value",
	"weight"
};

int lstm_strdef_get_id(const char* str)
{
	int i;
	int ret = LSTM_PARSE_FAILED;

	LOG("enter");

	for(i = 0; i < LSTM_STR_AMOUNT; i++)
	{
		ret = lstm_strcmp(str, lstm_str_list[i]);
		if(ret == LSTM_NO_ERROR)
		{
			ret = i;
			goto RET;
		}
	}

RET:
	LOG("exit");
	return ret;
}
