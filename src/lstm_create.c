#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lstm_private.h"
#include "lstm_builtin_math.h"

#include "debug.h"

#define DEFAULT_MAX_TIMESTEP	5

void lstm_zeromem(struct LSTM_STRUCT* lstm)
{
	memset(lstm, 0, sizeof(struct LSTM_STRUCT));
}

int lstm_create(lstm_t* lstmPtr, lstm_config_t lstmCfg)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_STRUCT* tmpLstmPtr = NULL;

	LOG("enter");

	// Memory allocation
	tmpLstmPtr = malloc(sizeof(struct LSTM_STRUCT));
	if(tmpLstmPtr == NULL)
	{
		ret = LSTM_MEM_FAILED;
		goto RET;
	}

	// Zero memory
	lstm_zeromem(tmpLstmPtr);

	// Clone config
	ret = lstm_config_struct_clone(&tmpLstmPtr->config, lstmCfg);
	if(ret != LSTM_NO_ERROR)
	{
		goto ERR;
	}

	// Allocate network
	ret = lstm_network_alloc(tmpLstmPtr, lstmCfg);
	if(ret != LSTM_NO_ERROR)
	{
		goto ERR;
	}

	// Set default max time step
	lstm_run(lstm_bptt_set_max_timestep(tmpLstmPtr, DEFAULT_MAX_TIMESTEP), ret, ERR);

	// Assign value
	*lstmPtr = tmpLstmPtr;

	goto RET;

ERR:
	lstm_struct_delete(tmpLstmPtr);
	lstm_free(tmpLstmPtr);

RET:
	LOG("exit");
	return ret;
}

