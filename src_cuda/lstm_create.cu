#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <debug.h>

#include "lstm_private_cuda.h"
#include "lstm_builtin_math_cuda.h"

int lstm_create_cuda(lstm_cuda_t* lstmCudaPtr, lstm_config_t lstmCfg)
{
	int ret = LSTM_NO_ERROR;
	struct LSTM_CUDA* tmpLstmPtr = NULL;
	void* allocTmp;

	LOG("enter");

	// Memory allocation
	lstm_alloc(allocTmp, 1, struct LSTM_CUDA, ret, RET);

	// Clone config
	lstm_run(lstm_config_struct_clone(&tmpLstmPtr->config, lstmCfg), ret, ERR);

	// Allocate network
	lstm_run(lstm_network_alloc_cuda(tmpLstmPtr, lstmCfg), ret, ERR);

	// Assing value
	*lstmCudaPtr = tmpLstmPtr;

	goto RET;

ERR:
	lstm_struct_delete_cuda(tmpLstmPtr);
	lstm_free(tmpLstmPtr);

RET:
	LOG("exit");
	return ret;
}
