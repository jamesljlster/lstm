#ifndef __LSTM_BUILTIN_MATH_CUDA_H__
#define __LSTM_BUILTIN_MATH_CUDA_H__

#ifdef __cplusplus
extern "C" {
#endif

__device__ void lstm_sigmoid_cu(double* dstPtr, double x);
__device__ void lstm_sigmoid_derivative_cu(double* dstPtr, double x);

__device__ void lstm_modified_sigmoid_cu(double* dstPtr, double x);
__device__ void lstm_modified_sigmoid_derivative_cu(double* dstPtr, double x);

__device__ void lstm_tanh_cu(double* dstPtr, double x);
__device__ void lstm_tanh_derivative_cu(double* dstPtr, double x);

__device__ void lstm_gaussian_cu(double* dstPtr, double x);
__device__ void lstm_gaussian_derivative_cu(double* dstPtr, double x);

__device__ void lstm_bent_identity_cu(double* dstPtr, double x);
__device__ void lstm_bent_identity_derivative_cu(double* dstPtr, double x);

__device__ void lstm_softplus_cu(double* dstPtr, double x);
__device__ void lstm_softplus_derivative_cu(double* dstPtr, double x);

__device__ void lstm_softsign_cu(double* dstPtr, double x);
__device__ void lstm_softsign_derivative_cu(double* dstPtr, double x);

__device__ void lstm_sinc_cu(double* dstPtr, double x);
__device__ void lstm_sinc_derivative_cu(double* dstPtr, double x);

__device__ void lstm_sinusoid_cu(double* dstPtr, double x);
__device__ void lstm_sinusoid_derivative_cu(double* dstPtr, double x);

__device__ void lstm_identity_cu(double* dstPtr, double x);
__device__ void lstm_identity_derivative_cu(double* dstPtr, double x);

__device__ void lstm_relu_cu(double* dstPtr, double x);
__device__ void lstm_relu_derivative_cu(double* dstPtr, double x);

#ifdef __cplusplus
}
#endif

__device__ void (*lstm_transfer_list[])(double*, double) = {
	lstm_sigmoid_cu,
	lstm_modified_sigmoid_cu,
	lstm_tanh_cu,
	lstm_gaussian_cu,
	lstm_bent_identity_cu,
	lstm_softplus_cu,
	lstm_softsign_cu,
	lstm_sinc_cu,
	lstm_sinusoid_cu,
	lstm_identity_cu,
	lstm_relu_cu
};

__device__ void (*lstm_transfer_derivative_list[])(double*, double) = {
	lstm_sigmoid_derivative_cu,
	lstm_modified_sigmoid_derivative_cu,
	lstm_tanh_derivative_cu,
	lstm_gaussian_derivative_cu,
	lstm_bent_identity_derivative_cu,
	lstm_softplus_derivative_cu,
	lstm_softsign_derivative_cu,
	lstm_sinc_derivative_cu,
	lstm_sinusoid_derivative_cu,
	lstm_identity_derivative_cu,
	lstm_relu_derivative_cu
};

#endif
