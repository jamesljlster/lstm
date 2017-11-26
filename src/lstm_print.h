#ifndef __LSTM_PRINT_H__
#define __LSTM_PRINT_H__

#include <stdio.h>

#include "lstm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void lstm_fprint_config(FILE* fptr, struct LSTM_CONFIG_STRUCT* lstmCfg, int indent);

void lstm_fprint_net(FILE* fptr, struct LSTM_STRUCT* lstm, int indent);
void lstm_fprint_layer(FILE* fptr, struct LSTM_STRUCT* lstm, int layerIndex, int indent);
void lstm_fprint_node(FILE* fptr, struct LSTM_STRUCT* lstm, int layerIndex, int nodeIndex, int indent);
void lstm_fprint_base(FILE* fptr, struct LSTM_BASE* basePtr, int netLen, int reNetSize, int indent);
void lstm_fprint_vector(FILE* fptr, float* vector, int vectorLen, int indent);

#ifdef __cplusplus
}
#endif

#endif
