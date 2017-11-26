#include <math.h>

#include "lstm.h"
#include "lstm_builtin_math.h"
#include "lstm_str.h"

#include "debug.h"

float (*lstm_transfer_list[])(float) = {
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

float (*lstm_transfer_derivative_list[])(float) = {
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

float lstm_sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

float lstm_sigmoid_derivative(float x)
{
    return lstm_sigmoid(x) * (1.0 - lstm_sigmoid(x));
}

float lstm_modified_sigmoid(float x)
{
	return 2.0 * lstm_sigmoid(x) - 1.0;
}

float lstm_modified_sigmoid_derivative(float x)
{
	return 2.0 * lstm_sigmoid_derivative(x);
}

float lstm_tanh(float x)
{
    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

float lstm_tanh_derivative(float x)
{
    return 1.0 - lstm_tanh(x) * lstm_tanh(x);
}

float lstm_gaussian(float x)
{
    return exp(-pow(x, 2) * 0.5);
}

float lstm_gaussian_derivative(float x)
{
    return -x * exp(-pow(x, 2) * 0.5);
}

float lstm_modified_gaussian(float x)
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

float lstm_modified_gaussian_derivative(float x)
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

float lstm_bent_identity(float x)
{
	return (sqrt(pow(x, 2) + 1.0) - 1) / 2.0 + x;
}

float lstm_bent_identity_derivative(float x)
{
	return x / (2.0 * sqrt(pow(x, 2) + 1)) + 1;
}

float lstm_softplus(float x)
{
	return log(1.0 + exp(x));
}

float lstm_softplus_derivative(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

float lstm_softsign(float x)
{
	return x / (1 + fabs(x));
}

float lstm_softsign_derivative(float x)
{
	return 1.0 / pow(1.0 + fabs(x), 2);
}

float lstm_sinc(float x)
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

float lstm_sinc_derivative(float x)
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

float lstm_sinusoid(float x)
{
	return sin(x);
}

float lstm_sinusoid_derivative(float x)
{
	return cos(x);
}

float lstm_identity(float x)
{
	return x;
}

float lstm_identity_derivative(float x)
{
	return 1;
}

float lstm_relu(float x)
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

float lstm_relu_derivative(float x)
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

