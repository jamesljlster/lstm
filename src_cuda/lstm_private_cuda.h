#ifndef __LSTM_PRIVATE_CUDA_H__
#define __LSTM_PRIVATE_CUDA_H__

#include <lstm_private.h>
#include "lstm_cuda.h"
#include "lstm_types_cuda.h"

// Macros
#ifdef DEBUG
#include <stdio.h>
#define lstm_free_cuda(ptr)	fprintf(stderr, "%s(): free(%s), %p\n", __FUNCTION__, #ptr, ptr); cudaFree(ptr)
#else
#define lstm_free_cuda(ptr)	cudaFree(ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Private allocate function
int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize);

// Private delete function
void lstm_struct_delete_cuda(struct LSTM_CUDA* lstmCuda);
void lstm_layer_delete_cuda(struct LSTM_CULAYER* cuLayerPtr);

#ifdef __cplusplus
}
#endif

#endif
