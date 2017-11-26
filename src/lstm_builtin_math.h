#ifndef __LSTM_BUILTIN_MATH_H__
#define __LSTM_BUILTIN_MATH_H__

// LSTM transfer function private definition
#define LSTM_TFUNC_AMOUNT	11

extern float (*lstm_transfer_list[])(float);
extern float (*lstm_transfer_derivative_list[])(float);
extern char* lstm_transfer_func_name[];

#ifdef __cplusplus
extern "C" {
#endif

float lstm_sigmoid(float x);
float lstm_sigmoid_derivative(float x);

float lstm_modified_sigmoid(float x);
float lstm_modified_sigmoid_derivative(float x);

float lstm_tanh(float x);
float lstm_tanh_derivative(float x);

float lstm_gaussian(float x);
float lstm_gaussian_derivative(float x);

float lstm_bent_identity(float x);
float lstm_bent_identity_derivative(float x);

float lstm_softplus(float x);
float lstm_softplus_derivative(float x);

float lstm_softsign(float x);
float lstm_softsign_derivative(float x);

float lstm_sinc(float x);
float lstm_sinc_derivative(float x);

float lstm_sinusoid(float x);
float lstm_sinusoid_derivative(float x);

float lstm_identity(float x);
float lstm_identity_derivative(float x);

float lstm_relu(float x);
float lstm_relu_derivative(float x);

#ifdef __cplusplus
}
#endif

#endif
