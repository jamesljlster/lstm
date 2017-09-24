#ifndef __LSTM_CUDA_H__
#define __LSTM_CUDA_H__

#include <lstm.h>

typedef struct LSTM_CUDA* lstm_cuda_t;

#ifdef __cplusplus
extern "C" {
#endif

int lstm_forward_computation_cuda(lstm_cuda_t lstmCuda, double* input, double* output);

int lstm_clone_to_cuda(lstm_cuda_t* lstmCudaPtr, lstm_t lstm);
int lstm_clone_from_cuda(lstm_t* lstmPtr, lstm_cuda_t lstmCuda);

void lstm_delete_cuda(lstm_cuda_t lstmCuda);

#ifdef __cplusplus
}
#endif

#endif
