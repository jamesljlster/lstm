#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	int iResult;
	lstm_config_t cfg;

	while(1)
	{
		iResult = lstm_config_create(&cfg);
		if(iResult < 0)
		{
			printf("lstm_config_create() failed with error: %d\n", iResult);
			return -1;
		}

		// Cleanup
		lstm_config_delete(cfg);
	}

	return 0;
}

