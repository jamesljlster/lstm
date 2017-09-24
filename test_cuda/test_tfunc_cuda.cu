#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#include <lstm_builtin_math_cuda.h>

int main()
{
	int i;
	double in, out, cuOut, err;

	cudaError_t cuErr;

	for(i = 0; i < LSTM_TFUNC_AMOUNT; i++)
	{
		err = 0;
		for(in = -1; in <= 1; in += 0.1)
		{
			out = lstm_transfer_list[i](in);
			run_tfunc<<<1, 1>>>(&cuOut, in, i);
			cudaDeviceSynchronize();

			err += fabs(out - cuOut);
		}

		printf("%s error: %lf\n", lstm_transfer_func_name[i], err);
		cuErr = cudaGetLastError();
		printf("Last cuda error: %s, %s\n", cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));
		printf("\n");
	}

	return 0;
}
