#ifndef __LSTM_STR_H__
#define __LSTM_STR_H__

struct LSTM_STR
{
	char* str;
	int strLen;
	int memLen;
};

#ifdef __cplusplus
extern "C" {
#endif

int lstm_str_append(struct LSTM_STR* strPtr, char ch);
int lstm_str_split(char*** strListPtr, int* strCountPtr, const char* src, const char* sepChs, const char* forceReadChs);

#ifdef __cpluspluc
}
#endif

#endif
