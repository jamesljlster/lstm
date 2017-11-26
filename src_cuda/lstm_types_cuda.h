#ifndef __LSTM_TYPES_CUDA_H__
#define __LSTM_TYPES_CUDA_H__

#include <lstm_types.h>

enum LSTM_CUMAT_LIST
{
	LSTM_CUMAT_INPUT,
	LSTM_CUMAT_OG,
	LSTM_CUMAT_FG,
	LSTM_CUMAT_IG,

	LSTM_CUMAT_AMOUNT
};

struct LSTM_CUMAT
{
	float* weight;
	float* wGrad;
	float* wDelta;

	float* calcBuf;

	float* calc;
	float* out;
	float* grad;
	float* gradHold;

	float* outQue;
	float* calcQue;
};

struct LSTM_CULAYER
{
	int inputTFunc;
	int outputTFunc;
	int gateTFunc;

	int nodeCount;
	int vecLen;

	float* output;
	float* cell;

	float* grad;

	float* outputQue;
	float* cellQue;

	struct LSTM_CUMAT baseMat;
};

struct LSTM_CUDA
{
	int queueLen;
	struct LSTM_CULAYER* layerList;

	struct LSTM_CONFIG_STRUCT config;
};

#endif
