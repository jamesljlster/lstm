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

int lstm_str_create(struct LSTM_STR* strPtr, const char* src);
int lstm_str_index_of(const char* src, char ch);
int lstm_str_append(struct LSTM_STR* strPtr, char ch);
int lstm_strcmp(const char* src1, const char* src2);
void lstm_str_trim(struct LSTM_STR* strPtr, const char* trimChs);

#ifdef __cpluspluc
}
#endif

#endif
