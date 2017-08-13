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

};

#endif
