#include <stdio.h>

#include "lstm.h"
#include "lstm_private.h"
#include "lstm_xml.h"
#include "lstm_file_proc.h"

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

