#ifndef __LSTM_PARSE_H__
#define __LSTM_PARSE_H__

#include "lstm_types.h"
#include "lstm_xml.h"

#ifdef __cplusplus
extern "C" {
#endif

int lstm_parse_config(struct LSTM_CONFIG_STRUCT* cfgPtr, struct LSTM_XML_ELEM* elemPtr);

#ifdef __cplusplus
}
#endif

#endif
