#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	int i;
	int iResult;
	lstm_config_t cfg, cpy;

	iResult = lstm_config_create(&cfg);
	if(iResult < 0)
	{
		printf("lstm_config_create() failed with error: %d\n", iResult);
		return -1;
	}

	while(1)
	{
		// Clone config
		iResult = lstm_config_clone(&cpy, cfg);
		if(iResult < 0)
		{
			printf("lstm_config_clone() failed with error: %d\n", iResult);
			return -1;
		}

		lstm_config_delete(cpy);
	}

	// Cleanup
	lstm_config_delete(cfg);

	return 0;
}

