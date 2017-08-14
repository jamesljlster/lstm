#include <stdlib.h>

#include "lstm.h"
#include "lstm_private.h"

void lstm_config_delete(lstm_config_t lstmCfg)
{
	lstm_config_delete_struct(lstmCfg);
	free(lstmCfg);
}

void lstm_config_delete_struct(struct LSTM_CONFIG_STRUCT* lstmCfgPtr)
{
	if(lstmCfgPtr->nodeList != NULL)
	{
		free(lstmCfgPtr->nodeList);
	}
}
