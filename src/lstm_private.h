#ifndef __LSTM_PRIVATE_H__
#define __LSTM_PRIVATE_H__

#include "lstm_types.h"

#ifdef DEBUG
#include <stdio.h>
#define lstm_free(ptr)	fprintf(stderr, "%s(): free(%s)\n", __FUNCTION__, #ptr); free(ptr)
#else
#define lstm_free(ptr)	free(ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Private config functions
void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfg);
int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfg);
void lstm_config_delete_struct(struct LSTM_CONFIG_STRUCT* lstmCfg);
int lstm_config_clone_struct(struct LSTM_CONFIG_STRUCT* dst, struct LSTM_CONFIG_STRUCT* src);

// Private delete functions
void lstm_buf_delete(struct LSTM_BUF* bufPtr);
void lstm_base_delete(struct LSTM_BASE* basePtr);
void lstm_node_delete(struct LSTM_NODE* nodePtr);
void lstm_layer_delete(struct LSTM_LAYER* layerPtr);
void lstm_delete_struct(struct LSTM_STRUCT* lstm);

// Private zero memory function
void lstm_zeromem(struct LSTM_STRUCT* lstm);

#ifdef __cplusplus
}
#endif

#endif
