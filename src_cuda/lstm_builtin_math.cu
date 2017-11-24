
#include <cuda_runtime.h>

#include "lstm_builtin_math_cuda.h"

__global__ void run_tfunc(double* out, double in, int tFuncIndex)
{
	lstm_transfer_list_cu[tFuncIndex](out, in);
}

__global__ void run_tfunc_de(double* out, double in, int tFuncIndex)
{
	lstm_transfer_derivative_list_cu[tFuncIndex](out, in);
}

__device__ void (*lstm_transfer_list_cu[])(double*, double) = {
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

__device__ void (*lstm_transfer_derivative_list_cu[])(double*, double) = {
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

__device__ void lstm_sigmoid_cu(double* dstPtr, double x)
{
	*dstPtr = 1.0 / (1.0 + exp(-x));
}

__device__ void lstm_sigmoid_derivative_cu(double* dstPtr, double x)
{
	double tmp;

	lstm_sigmoid_cu(&tmp, x);
	*dstPtr = tmp * (1.0 - tmp);
}

__device__ void lstm_modified_sigmoid_cu(double* dstPtr, double x)
{
	double tmp;

	lstm_sigmoid_cu(&tmp, x);
	*dstPtr = 2.0 * tmp - 1.0;
}

__device__ void lstm_modified_sigmoid_derivative_cu(double* dstPtr, double x)
{
	double tmp;

	lstm_sigmoid_derivative_cu(&tmp, x);
	*dstPtr = 2.0 * tmp;
}

__device__ void lstm_tanh_cu(double* dstPtr, double x)
{
	*dstPtr = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

__device__ void lstm_tanh_derivative_cu(double* dstPtr, double x)
{
	double tmp;

	lstm_tanh_cu(&tmp, x);
	*dstPtr = 1.0 - tmp * tmp;
}

__device__ void lstm_gaussian_cu(double* dstPtr, double x)
{
	*dstPtr = exp(-pow(x, 2) * 0.5);
}

__device__ void lstm_gaussian_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = -x * exp(-pow(x, 2) * 0.5);
}

__device__ void lstm_modified_gaussian_cu(double* dstPtr, double x)
{
	if(x == 0)
	{
		*dstPtr = 1;
	}
	else
	{
		*dstPtr = sin(x) / x;
	}
}

__device__ void lstm_modified_gaussian_derivative_cu(double* dstPtr, double x)
{
	if(x == 0)
	{
		*dstPtr = 0;
	}
	else
	{
		*dstPtr = (cos(x) / x) - (sin(x) / pow(x, 2));
	}
}

__device__ void lstm_bent_identity_cu(double* dstPtr, double x)
{
	*dstPtr = (sqrt(pow(x, 2) + 1.0) - 1) / 2.0 + x;
}

__device__ void lstm_bent_identity_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = x / (2.0 * sqrt(pow(x, 2) + 1)) + 1;
}

__device__ void lstm_softplus_cu(double* dstPtr, double x)
{
	*dstPtr = log(1.0 + exp(x));
}

__device__ void lstm_softplus_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = 1.0 / (1.0 + exp(-x));
}

__device__ void lstm_softsign_cu(double* dstPtr, double x)
{
	*dstPtr = x / (1 + fabs(x));
}

__device__ void lstm_softsign_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = 1.0 / pow(1.0 + fabs(x), 2);
}

__device__ void lstm_sinc_cu(double* dstPtr, double x)
{
	if(x == 0.0)
	{
		*dstPtr = 1.0;
	}
	else
	{
		*dstPtr = sin(x) / x;
	}
}

__device__ void lstm_sinc_derivative_cu(double* dstPtr, double x)
{
	if(x == 0.0)
	{
		*dstPtr = 0.0;
	}
	else
	{
		*dstPtr = (cos(x) / x) - (sin(x) / pow(x, 2));
	}
}

__device__ void lstm_sinusoid_cu(double* dstPtr, double x)
{
	*dstPtr = sin(x);
}

__device__ void lstm_sinusoid_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = cos(x);
}

__device__ void lstm_identity_cu(double* dstPtr, double x)
{
	*dstPtr = x;
}

__device__ void lstm_identity_derivative_cu(double* dstPtr, double x)
{
	*dstPtr = 1;
}

__device__ void lstm_relu_cu(double* dstPtr, double x)
{
	if(x < 0.0)
	{
		*dstPtr = 0;
	}
	else
	{
		*dstPtr = x;
	}
}

__device__ void lstm_relu_derivative_cu(double* dstPtr, double x)
{
	if(x < 0.0)
	{
		*dstPtr = 0;
	}
	else
	{
		*dstPtr = 1;
	}
}
