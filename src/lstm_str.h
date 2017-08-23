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
int lstm_strcmp(const char* src1, const char* src2);

#ifdef __cpluspluc
}
#endif

#endif
