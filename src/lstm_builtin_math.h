#ifndef __LSTM_BUILTIN_MATH_H__
#define __LSTM_BUILTIN_MATH_H__

// LSTM transfer function private definition
#define LSTM_TFUNC_AMOUNT	11

extern double (*lstm_transfer_list[])(double);
extern double (*lstm_transfer_derivative_list[])(double);
extern char* lstm_transfer_func_name[];

#ifdef __cplusplus
extern "C" {
#endif

double lstm_sigmoid(double x);
double lstm_sigmoid_derivative(double x);

double lstm_modified_sigmoid(double x);
double lstm_modified_sigmoid_derivative(double x);

double lstm_tanh(double x);
double lstm_tanh_derivative(double x);

double lstm_gaussian(double x);
double lstm_gaussian_derivative(double x);

double lstm_bent_identity(double x);
double lstm_bent_identity_derivative(double x);

double lstm_softplus(double x);
double lstm_softplus_derivative(double x);

double lstm_softsign(double x);
double lstm_softsign_derivative(double x);

double lstm_sinc(double x);
double lstm_sinc_derivative(double x);

double lstm_sinusoid(double x);
double lstm_sinusoid_derivative(double x);

double lstm_identity(double x);
double lstm_identity_derivative(double x);

double lstm_relu(double x);
double lstm_relu_derivative(double x);

#ifdef __cplusplus
}
#endif

#endif
