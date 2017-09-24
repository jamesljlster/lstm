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
	double* weight;
	double* wGrad;
	double* wDelta;

	double* calcBuf;

	double* calc;
	double* out;
	double* grad;
	double* gradHold;

	double* outQue;
	double* calcQue;
};

struct LSTM_CULAYER
{
	void (*inputTFunc)(double*, double);
	void (*inputDTFunc)(double*, double);

	void (*outputTFunc)(double*, double);
	void (*outputDTFunc)(double*, double);

	void (*gateTFunc)(double*, double);
	void (*gateDTFunc)(double*, double);

	int nodeCount;
	int vecLen;

	double* output;
	double* cell;

	double* grad;

	double* outputQue;
	double* cellQue;

	struct LSTM_CUMAT baseMat;
};

struct LSTM_CUDA
{
	int queueLen;
	struct LSTM_CULAYER* layerList;

	struct LSTM_CONFIG_STRUCT config;
};

#endif
