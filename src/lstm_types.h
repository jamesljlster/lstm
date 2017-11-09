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

	double calc;		// Calcalution temp: value before activation function
	double out;			// Network output value
	double grad;		// Gradient value
	double gradHold;	// Gradient value backup

	struct LSTM_BUF outQue;		// out queue
	struct LSTM_BUF calcQue;	// calc queue
};

struct LSTM_NODE
{
	struct LSTM_BASE ogNet;		// Output gate network
	struct LSTM_BASE fgNet;		// Forget gate network
	struct LSTM_BASE igNet;		// Input gate network
	struct LSTM_BASE inputNet;	// Input network

	double output;				// Output of node
	double cell;				// Cell value of node

	struct LSTM_BUF outputQue;	// Output queue
	struct LSTM_BUF cellQue;	// Cell value queue

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

struct LSTM_STATE_STRUCT
{
	double* cell;	// Cell matrix of lstm blocks
	double* hidden;	// Hidden state matrix of lstm blocks

	struct LSTM_CONFIG_STRUCT config;	// Store network configure for verifying.
};

struct LSTM_STRUCT
{
	//int queueLen;
	int queueSize;
	int queueHead;
	int queueTail;

	struct LSTM_LAYER* layerList;

	struct LSTM_CONFIG_STRUCT config;
};

#endif
