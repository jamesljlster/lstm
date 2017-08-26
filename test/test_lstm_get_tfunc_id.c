#include <stdio.h>

#include "lstm.h"
#include "lstm_builtin_math.h"

int main()
{
	int i;

	const char* tmp;

	for(i = 0; i < LSTM_TFUNC_AMOUNT; i++)
	{
		tmp = lstm_transfer_func_name[i];
		printf("%s: %d\n", tmp, lstm_get_transfer_func_id(tmp));
	}

	return 0;
}
