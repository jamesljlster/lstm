#ifndef __LSTM_PRIVATE_CUDA_H__
#define __LSTM_PRIVATE_CUDA_H__

#include <lstm_private.h>
#include "lstm_cuda.h"
#include "lstm_types_cuda.h"

// Macros
#ifdef DEBUG
#include <stdio.h>
#define lstm_free_cuda(ptr)	fprintf(stderr, "%s(): cudaFree(%s), %p\n", __FUNCTION__, #ptr, ptr); cudaFree(ptr)
#else
#define lstm_free_cuda(ptr)	cudaFree(ptr)
#endif

#define lstm_alloc_cuda(var, len, type, retVar, errLabel) \
	if(cudaMalloc(&var, len * sizeof(type)) != cudaSuccess) \
	{ \
		retVar = LSTM_MEM_FAILED; \
		goto errLabel; \
	}

#ifdef __cplusplus
extern "C" {
#endif

// Private allocate function
int lstm_network_alloc_cuda(struct LSTM_CUDA* lstmCuda, const struct LSTM_CONFIG_STRUCT* lstmCfg);
int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize);

// Private delete function
void lstm_struct_delete_cuda(struct LSTM_CUDA* lstmCuda);
void lstm_layer_delete_cuda(struct LSTM_CULAYER* cuLayerPtr);

#ifdef __cplusplus
}
#endif

#endif
