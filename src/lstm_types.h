#ifndef __LSTM_TYPES_H__
#define __LSTM_TYPES_H__

struct LSTM_BUF
{
	float* list;
	int listLen;
};

struct LSTM_BASE
{
	float* weight;		// Weight connection
	float* wGrad;		// Gradient summation of weight
	float* wDelta;		// Momentum of weight

	float* rWeight;	// Recurrent weight connection
	float* rGrad;		// Gradient summation of recurrent weight
	float* rDelta;		// Momentum of recurrent weight

	float th;			// Threshold of the node
	float thGrad;		// Gradient summation of threshold
	float thDelta;		// Momentum of threshold

	float calc;		// Calcalution temp: value before activation function
	float out;			// Network output value
	float grad;		// Gradient value
	float gradHold;	// Gradient value backup

	struct LSTM_BUF outQue;		// out queue
	struct LSTM_BUF calcQue;	// calc queue
};

struct LSTM_NODE
{
	struct LSTM_BASE ogNet;		// Output gate network
	struct LSTM_BASE fgNet;		// Forget gate network
	struct LSTM_BASE igNet;		// Input gate network
	struct LSTM_BASE inputNet;	// Input network

	float output;				// Output of node
	float cell;				// Cell value of node

	struct LSTM_BUF outputQue;	// Output queue
	struct LSTM_BUF cellQue;	// Cell value queue

	float grad;	// Node gradient, for bp calculation
	float rHold;	// For recurrent forward computation
};

struct LSTM_LAYER
{
	float (*inputTFunc)(float);	// Input transfer function
	float (*inputDTFunc)(float);	// Input derivative transfer function

	float (*outputTFunc)(float);	// Output transfer function
	float (*outputDTFunc)(float);	// Output derivative transfer function

	float (*gateTFunc)(float);	// Gate transfer function
	float (*gateDTFunc)(float);	// Gate derivative transfer function

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

	float lRate;
	float mCoef;

	int* nodeList;
};

struct LSTM_STATE_STRUCT
{
	float** cell;		// Cell vector list
	float* hidden;		// Hidden state vector list

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
