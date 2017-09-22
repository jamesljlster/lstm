#ifndef __LSTM_PRIVATE_CUDA_H__
#define __LSTM_PRIVATE_CUDA_H__

#include <lstm_private.h>
#include "lstm_types_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private allocate function
int lstm_layer_alloc_cuda(struct LSTM_CULAYER* cuLayerPtr, int nodeCount, int nodeType, int netSize, int reNetSize);

#ifdef __cplusplus
}
#endif

#endif
