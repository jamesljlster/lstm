#include <string.h>

#include "lstm.h"
#include "lstm_private.h"

void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	memset(lstmCfgPtr, 0, sizeof(struct LSTM_CONFIG_STRUCT));
}
