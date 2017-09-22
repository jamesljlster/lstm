#ifndef __LSTM_TYPES_CUDA_H__
#define __LSTM_TYPES_CUDA_H__

#include <lstm_types.h>

enum LSTM_CUMAT_LIST
{
	LSTM_CUMAT_OG,
	LSTM_CUMAT_FG,
	LSTM_CUMAT_IG,
	LSTM_CUMAT_INPUT,

	LSTM_CUMAT_AMOUNT
};

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
	int inputTFuncIndex;
	int outputTFuncIndex;
	int gateTFuncIndex;

	int nodeCount;
	int vecLen;

	struct LSTM_CUMAT nodeMat;
};

struct LSTM_CUSTRUCT
{
	int queueLen;
	struct LSTM_CULAYER* layerList;

	struct LSTM_CONFIG_STRUCT config;
};

#endif
