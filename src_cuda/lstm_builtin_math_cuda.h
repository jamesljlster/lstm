#ifndef __LSTM_BUILTIN_MATH_CUDA_H__
#define __LSTM_BUILTIN_MATH_CUDA_H__

#include <lstm_builtin_math.h>

extern __device__ void (*lstm_transfer_list_cu[])(float*, float);
extern __device__ void (*lstm_transfer_derivative_list_cu[])(float*, float);

#ifdef __cplusplus
extern "C" {
#endif

__global__ void run_tfunc(float* out, float in, int tFuncIndex);
__global__ void run_tfunc_de(float* out, float in, int tFuncIndex);

__device__ void lstm_sigmoid_cu(float* dstPtr, float x);
__device__ void lstm_sigmoid_derivative_cu(float* dstPtr, float x);
__device__ void lstm_modified_sigmoid_cu(float* dstPtr, float x);
__device__ void lstm_modified_sigmoid_derivative_cu(float* dstPtr, float x);
__device__ void lstm_tanh_cu(float* dstPtr, float x);
__device__ void lstm_tanh_derivative_cu(float* dstPtr, float x);
__device__ void lstm_gaussian_cu(float* dstPtr, float x);
__device__ void lstm_gaussian_derivative_cu(float* dstPtr, float x);
__device__ void lstm_modified_gaussian_cu(float* dstPtr, float x);
__device__ void lstm_modified_gaussian_derivative_cu(float* dstPtr, float x);
__device__ void lstm_bent_identity_cu(float* dstPtr, float x);
__device__ void lstm_bent_identity_derivative_cu(float* dstPtr, float x);
__device__ void lstm_softplus_cu(float* dstPtr, float x);
__device__ void lstm_softplus_derivative_cu(float* dstPtr, float x);
__device__ void lstm_softsign_cu(float* dstPtr, float x);
__device__ void lstm_softsign_derivative_cu(float* dstPtr, float x);
__device__ void lstm_sinc_cu(float* dstPtr, float x);
__device__ void lstm_sinc_derivative_cu(float* dstPtr, float x);
__device__ void lstm_sinusoid_cu(float* dstPtr, float x);
__device__ void lstm_sinusoid_derivative_cu(float* dstPtr, float x);
__device__ void lstm_identity_cu(float* dstPtr, float x);
__device__ void lstm_identity_derivative_cu(float* dstPtr, float x);
__device__ void lstm_relu_cu(float* dstPtr, float x);
__device__ void lstm_relu_derivative_cu(float* dstPtr, float x);

#ifdef __cplusplus
}
#endif

#endif
