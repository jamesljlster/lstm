#ifndef __LSTM_PRINT_H__
#define __LSTM_PRINT_H__

#include <stdio.h>

#include "lstm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void lstm_fprint_config(FILE* fptr, struct LSTM_CONFIG_STRUCT* lstmCfg, int indent);

#ifdef __cplusplus
}
#endif

#endif
