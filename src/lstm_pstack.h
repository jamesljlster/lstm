#ifndef __LSTM_PSTACK_H__
#define __LSTM_PSTACK_H__

struct LSTM_PSTACK
{
	void** stack;
	int stackSize;
	int top;
};

#ifdef __cplusplus
extern "C" {
#endif

void lstm_pstack_init(struct LSTM_PSTACK* pstack);

int lstm_pstack_push(struct LSTM_PSTACK* pstack, void* ptr);
void* lstm_pstack_pop(struct LSTM_PSTACK* pstack);

void lstm_pstack_delete(struct LSTM_PSTACK* pstack);

#ifdef __cplusplus
}
#endif

#endif
