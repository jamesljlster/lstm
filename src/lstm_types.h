#ifndef __LSTM_TYPES_H__
#define __LSTM_TYPES_H__

struct LSTM_BUF
{
	double* list;
	int listLen;
};

struct LSTM_BASE
{
	double* weight;		// Weight connection
	double* wGrad;		// Gradient summation of weight
	double* wDelta;		// Momentum of weight

	double* rWeight;	// Recurrent weight connection
	double* rGrad;		// Gradient summation of recurrent weight
	double* rDelta;		// Momentum of recurrent weight

	double th;			// Threshold of the node
	double thGrad;		// Gradient summation of threshold
	double thDelta;		// Momentum of threshold
};

struct LSTM_NODE
{
	struct LSTM_BASE ogNet;		// Output gate network
	struct LSTM_BASE fgNet;		// Forget gate network
	struct LSTM_BASE igNet;		// Input gate network
	struct LSTM_BASE inputNet;	// Input network

	struct LSTM_BUF outputQue;	// Output queue
	struct LSTM_BUF cellQue;	// Cell value queue

	struct LSTM_BUF ogQue;		// Output gate queue
	struct LSTM_BUF ogCalcQue;	// Output gate temp calculation queue

	struct LSTM_BUF fgQue;		// Forget gate queue
	struct LSTM_BUF fgCalcQue;	// Forget gate temp calculation queue

	struct LSTM_BUF igQue;		// Input gate queue
	struct LSTM_BUF igCalcQue;	// Input gate temp calculation queue

	double grad;	// Node gradient, for bp calculation
	double rHold;	// For recurrent forward computation
};

struct LSTM_LAYER
{
	double (*inputTFunc)(double);	// Input transfer function
	double (*inputDTFunc)(double);	// Input derivative transfer function

	double (*outputTFunc)(double);	// Output transfer function
	double (*outputDTFunc)(double);	// Output derivative transfer function

	double (*gateTFunc)(double);	// Gate transfer function
	double (*gateDTFunc)(double);	// Gate derivative transfer function

	struct LSTM_NODE* nodeList;
	int nodeCount;
};

struct LSTM_CONFIG_STRUCT
{
	int inputs;
	int outputs;
	int layers;

	int inputTFunc;
	int outputTFunc;
	int gateTFunc;

	double lRate;
	double mCoef;

	int* nodeList;
};

struct LSTM_STRUCT
{
	int queueLen;
	struct LSTM_LAYER* layerList;

	struct LSTM_CONFIG_STRUCT config;
};

#endif
