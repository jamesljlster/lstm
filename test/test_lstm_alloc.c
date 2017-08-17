#include <stdio.h>
#include <stdlib.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	void* ptr;
	int ret;

	unsigned int bigSize = 0xffffffff;
	unsigned int smallSize = 100;

	// Test allocate
	lstm_alloc(ptr, smallSize, double, ret, ERR);
	printf("Memory allocated\n");

	lstm_alloc(ptr, bigSize, double, ret, ERR);
	printf("Memory allocated\n");

	goto RET;

ERR:
	printf("Memory allocation failed!\n");

RET:
	return 0;
}
