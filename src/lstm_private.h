#ifndef __LSTM_PRIVATE_H__
#define __LSTM_PRIVATE_H__

#include "lstm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfgPtr);
int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfgPtr);
void lstm_config_delete_struct(struct LSTM_CONFIG_STRUCT* lstmCfgPtr);

#ifdef __cplusplus
}
#endif

#endif
