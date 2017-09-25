#ifndef __LSTM_DUMP_CUDA_H__
#define __LSTM_DUMP_CUDA_H__

#ifdef DUMP_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define DUMP(msg, ptr, len) \
{ \
	double* dumpTmp = (double*)calloc(len, sizeof(double)); \
	if(dumpTmp == NULL) \
	{ \
		fprintf(stderr, "%s(): Memory allocation failed while trying to dump %s with length %d\n", __FUNCTION__, #ptr, len); \
	} \
	else \
	{ \
		cudaError_t cuErr = cudaMemcpy(dumpTmp, ptr, len * sizeof(double), cudaMemcpyDeviceToHost); \
		if(cuErr != cudaSuccess) \
		{ \
			fprintf(stderr, "%s(): cudaMemcpy() failed while trying to dump %s with length %d\n", __FUNCTION__, #ptr, len); \
		} \
		else \
		{ \
			int i; \
			fprintf(stderr, "%s(): %s", __FUNCTION__, msg); \
			for(i = 0; i < len; i++) \
			{ \
				fprintf(stderr, "%lf", dumpTmp[i]); \
				if(i == len - 1) \
				{ \
					fprintf(stderr, "\n"); \
				} \
				else \
				{ \
					fprintf(stderr, ", "); \
				} \
			} \
		} \
	} \
	free(dumpTmp); \
}
#else
#define DUMP(msg, ptr, len)
#endif

#endif
