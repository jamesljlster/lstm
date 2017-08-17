#ifndef __LSTM_PRIVATE_H__
#define __LSTM_PRIVATE_H__

#include <stdlib.h>

#include "lstm.h"
#include "lstm_types.h"

// Private definitions
enum LSTM_NODE_TYPE
{
	LSTM_FULL_NODE,
	LSTM_OUTPUT_NODE
};

// Macros
#ifdef DEBUG
#include <stdio.h>
#define lstm_free(ptr)	fprintf(stderr, "%s(): free(%s), %p\n", __FUNCTION__, #ptr, ptr); free(ptr)
#else
#define lstm_free(ptr)	free(ptr)
#endif

#define lstm_alloc(ptr, len, type, retVar, errLabel) \
	ptr = calloc(len, sizeof(type)); \
	if(ptr == NULL) \
	{ \
		ret = LSTM_MEM_FAILED; \
		goto errLabel; \
	}

#ifdef __cplusplus
extern "C" {
#endif

// Private config functions
void lstm_config_zeromem(struct LSTM_CONFIG_STRUCT* lstmCfg);
int lstm_config_init(struct LSTM_CONFIG_STRUCT* lstmCfg);
void lstm_config_struct_delete(struct LSTM_CONFIG_STRUCT* lstmCfg);
int lstm_config_struct_clone(struct LSTM_CONFIG_STRUCT* dst, struct LSTM_CONFIG_STRUCT* src);

// Private delete functions
void lstm_buf_delete(struct LSTM_BUF* bufPtr);
void lstm_base_delete(struct LSTM_BASE* basePtr);
void lstm_node_delete(struct LSTM_NODE* nodePtr);
void lstm_layer_delete(struct LSTM_LAYER* layerPtr);
void lstm_struct_delete(struct LSTM_STRUCT* lstm);

// Private zero memory function
void lstm_zeromem(struct LSTM_STRUCT* lstm);

#ifdef __cplusplus
}
#endif

#endif
