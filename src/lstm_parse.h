#ifndef __LSTM_PARSE_H__
#define __LSTM_PARSE_H__

#include "lstm.h"
#include "lstm_types.h"
#include "lstm_xml.h"

#ifdef __cplusplus
extern "C" {
#endif

int lstm_parse_network_xml(struct LSTM_STRUCT* lstm, struct LSTM_XML_ELEM* elemPtr);

int lstm_parse_config_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_config_tfun_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);
int lstm_parse_config_hidden_layer_xml(struct LSTM_CONFIG_STRUCT* lstmCfg, struct LSTM_XML_ELEM* elemPtr);

#ifdef __cplusplus
}
#endif

#endif
