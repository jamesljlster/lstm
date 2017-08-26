#include <stdio.h>

#include <lstm.h>
#include <lstm_private.h>

int main()
{
	int i;
	int iResult;
	lstm_config_t cfg;

	lstm_run(lstm_config_create(&cfg), iResult, RET);
	lstm_run(lstm_config_set_inputs(cfg, -1), iResult, RET);

	// Cleanup
	lstm_config_delete(cfg);

RET:
	return 0;
}

