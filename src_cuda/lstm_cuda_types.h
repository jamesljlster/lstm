#ifndef __LSTM_CUDA_TYPES_H__
#define __LSTM_CUDA_TYPES_H__

struct LSTM_CUDA_MAT
{
	double* weight;
	double* wGrad;
	double* wDelta;

	double* th;
	double* thGrad;
	double* thDelta;

	double* calc;
	double* out;
	double* grad;
	double* gradHold;

	double* outQue;
	double* calcQue;
};

struct LSTM_CUDA_LAYER
{
	int nodeCount;

	struct LSTM_CUDA_MAT nodeMat;
};

#endif
