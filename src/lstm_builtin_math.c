#include <math.h>

#include "lstm.h"
#include "lstm_builtin_math.h"
#include "lstm_str.h"

#include "debug.h"

double (*lstm_transfer_list[])(double) = {
	lstm_sigmoid,
	lstm_modified_sigmoid,
	lstm_tanh,
	lstm_gaussian,
	lstm_bent_identity,
	lstm_softplus,
	lstm_softsign,
	lstm_sinc,
	lstm_sinusoid,
	lstm_identity,
	lstm_relu
};

double (*lstm_transfer_derivative_list[])(double) = {
	lstm_sigmoid_derivative,
	lstm_modified_sigmoid_derivative,
	lstm_tanh_derivative,
	lstm_gaussian_derivative,
	lstm_bent_identity_derivative,
	lstm_softplus_derivative,
	lstm_softsign_derivative,
	lstm_sinc_derivative,
	lstm_sinusoid_derivative,
	lstm_identity_derivative,
	lstm_relu_derivative
};

char* lstm_transfer_func_name[] = {
	"Sigmoid",
	"Mod. Sigmoid",
	"Hyperbolic Tangent",
	"Gaussian",
	"Bent Identity",
	"SoftPlus",
	"SoftSign",
	"Sinc",
	"Sinusoid",
	"Identity",
	"Rectifier Linear Unit"
};

int lstm_get_transfer_func_id(const char* tFuncName)
{
	int i;
	int ret = LSTM_PARSE_FAILED;

	LOG("enter");

	for(i = 0; i < LSTM_TFUNC_AMOUNT; i++)
	{
		ret = lstm_strcmp(tFuncName, lstm_transfer_func_name[i]);
		if(ret == LSTM_NO_ERROR)
		{
			ret = i;
			goto RET;
		}
	}

RET:
	LOG("exit");
	return ret;
}

double lstm_sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double lstm_sigmoid_derivative(double x)
{
    return lstm_sigmoid(x) * (1.0 - lstm_sigmoid(x));
}

double lstm_modified_sigmoid(double x)
{
	return 2.0 * lstm_sigmoid(x) - 1.0;
}

double lstm_modified_sigmoid_derivative(double x)
{
	return 2.0 * lstm_sigmoid_derivative(x);
}

double lstm_tanh(double x)
{
    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

double lstm_tanh_derivative(double x)
{
    return 1.0 - lstm_tanh(x) * lstm_tanh(x);
}

double lstm_gaussian(double x)
{
    return exp(-pow(x, 2) * 0.5);
}

double lstm_gaussian_derivative(double x)
{
    return -x * exp(-pow(x, 2) * 0.5);
}

double lstm_modified_gaussian(double x)
{
	if(x == 0)
	{
		return 1;
	}
	else
	{
		return sin(x) / x;
	}
}

double lstm_modified_gaussian_derivative(double x)
{
	if(x == 0)
	{
		return 0;
	}
	else
	{
		return (cos(x) / x) - (sin(x) / pow(x, 2));
	}
}

double lstm_bent_identity(double x)
{
	return (sqrt(pow(x, 2) + 1.0) - 1) / 2.0 + x;
}

double lstm_bent_identity_derivative(double x)
{
	return x / (2.0 * sqrt(pow(x, 2) + 1)) + 1;
}

double lstm_softplus(double x)
{
	return log(1.0 + exp(x));
}

double lstm_softplus_derivative(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double lstm_softsign(double x)
{
	return x / (1 + fabs(x));
}

double lstm_softsign_derivative(double x)
{
	return 1.0 / pow(1.0 + fabs(x), 2);
}

double lstm_sinc(double x)
{
	if(x == 0.0)
	{
		return 1.0;
	}
	else
	{
		return sin(x) / x;
	}
}

double lstm_sinc_derivative(double x)
{
	if(x == 0.0)
	{
		return 0.0;
	}
	else
	{
		return (cos(x) / x) - (sin(x) / pow(x, 2));
	}
}

double lstm_sinusoid(double x)
{
	return sin(x);
}

double lstm_sinusoid_derivative(double x)
{
	return cos(x);
}

double lstm_identity(double x)
{
	return x;
}

double lstm_identity_derivative(double x)
{
	return 1;
}

double lstm_relu(double x)
{
	if(x < 0.0)
	{
		return 0;
	}
	else
	{
		return x;
	}
}

double lstm_relu_derivative(double x)
{
	if(x < 0.0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

