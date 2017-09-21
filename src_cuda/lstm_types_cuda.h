#ifndef __LSTM_TYPES_CUDA_H__
#define __LSTM_TYPES_CUDA_H__

struct LSTM_CUMAT
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

struct LSTM_CULAYER
{
	int nodeCount;

	struct LSTM_CUMAT nodeMat;
};

#endif
