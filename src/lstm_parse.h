#ifndef __LSTM_PARSE_H__
#define __LSTM_PARSE_H__

#include "lstm.h"
#include "lstm_types.h"
#include "lstm_xml.h"

#ifdef __cplusplus
extern "C" {
#endif

int lstm_parse_net_xml(struct LSTM_STRUCT* lstm, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_net_layer_xml(struct LSTM_STRUCT* lstm, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_net_node_xml(struct LSTM_STRUCT* lstm, int layerIndex, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_net_base_xml(struct LSTM_BASE* basePtr, int netLen, int layerIndex, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_net_vector_xml(double* vector, int vectorLen, struct LSTM_XML_ELEM* elemPtr);

int lstm_parse_config_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_config_tfun_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_config_hidden_layer_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_config_hidden_nodes_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);

#ifdef __cplusplus
}
#endif

#endif
