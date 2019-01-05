/**
 *	@author	Zheng-Ling Lai <jamesljlster@gmail.com>
 *	@file	lstm.h
**/

#ifndef __LSTM_H__
#define __LSTM_H__

/**
 *	\~English @defgroup Enum Enumeration
 *	@brief Enumeration of the Library.
 *	@since 0.1.0
 *
 * @{
 *
 *	\~Chinese-Traditional @defgroup Enum 列舉
 *	@brief 函式庫列舉
 *	@since 0.1.0
 *
 * @{
 */

/**
 *	\~English
 *	@brief Return value definitions of LSTM library.
 *	@since 0.1.0
 *
 *	\~Chinese-Traditional
 *	@brief LSTM 函式庫回傳值定義
 *	@since 0.1.0
 */
enum LSTM_RETUEN_VALUE
{
	LSTM_NO_ERROR		= 0,	/*!< No error occured while running called function. @since 0.1.0 */
	LSTM_MEM_FAILED		= -1,	/*!< Memory allocation failed. @since 0.1.0 */
	LSTM_INVALID_ARG	= -2,	/*!< Invalid argument(s) or setting(s). @since 0.1.0 */
	LSTM_FILE_OP_FAILED	= -3,	/*!< File operation failed. @since 0.1.0 */
	LSTM_PARSE_FAILED	= -4,	/*!< Error occurred while parsing file. @since 0.1.0 */
	LSTM_OUT_OF_RANGE	= -5,	/*!< Operation out of range. @since 0.1.0 */
	LSTM_NOT_MATCH		= -6	/*!< Comparison not match. @since 0.1.0 */
};

/**
 *	\~English
 *	@brief Transfer (activation) function index definitions.
 *	@since 0.1.0
 *
 *	\~Chinese-Traditional
 *	@brief 活化函數定義
 *	@since 0.1.0
 */
enum LSTM_TRANSFER_FUNC
{
	LSTM_SIGMOID			= 0,	/*!< Sigmoid function. @since 0.1.0 */
	LSTM_MODIFIED_SIGMOID	= 1,	/*!< Modified sigmoid function. @since 0.1.0 */
	LSTM_HYPERBOLIC_TANGENT	= 2,	/*!< Hyperbolic tangent function. @since 0.1.0 */
	LSTM_GAUSSIAN			= 3,	/*!< Gaussian function. @since 0.1.0 */
	LSTM_BENT_IDENTITY		= 4,	/*!< Bent identity function. @since 0.1.0 */
	LSTM_SOFTPLUS			= 5,	/*!< SoftPlus function. @since 0.1.0 */
	LSTM_SOFTSIGN			= 6,	/*!< SoftSign function. @since 0.1.0 */
	LSTM_SINC				= 7,	/*!< Sinc function. @since 0.1.0 */
	LSTM_SINUSOID			= 8,	/*!< Sinusoid (sine) function. @since 0.1.0 */
	LSTM_IDENTITY			= 9,	/*!< Identity function. @since 0.1.0 */
	LSTM_RELU				= 10	/*!< Rectifier linear unit function. @since 0.1.0 */
};

/**
 * @}
 */

/**
 *	\~English @defgroup Types Data Types
 *	@brief Data Types of the Library.
 *
 * @{
 *
 *	\~Chinese-Traditional @defgroup Types 資料型別
 *	@brief 函式庫資料型別
 *
 * @{
 */

/** Type definition of lstm. @since 0.1.0 */
typedef struct LSTM_STRUCT* lstm_t;

/** Type definition of lstm configuration. @since 0.1.0 */
typedef struct LSTM_CONFIG_STRUCT* lstm_config_t;

/** Type definition of lstm state. @since 0.1.0 */
typedef struct LSTM_STATE_STRUCT* lstm_state_t;

/**
 * @}
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 *	\~English @defgroup Config Configuration
 *	@brief Configuration of LSTM networks.
 *
 * @{
 *
 *	\~Chinese-Traditional @defgroup Config 配置
 *	@brief LSTM 網路配置
 *
 * @{
 */

int lstm_config_create(lstm_config_t* lstmCfgPtr);
int lstm_config_clone(lstm_config_t* lstmCfgPtr, const lstm_config_t lstmCfgSrc);
int lstm_config_arch_compare(lstm_config_t src1, lstm_config_t src2);
void lstm_config_delete(lstm_config_t lstmCfg);
int lstm_config_set_inputs(lstm_config_t lstmCfg, int inputs);
int lstm_config_get_inputs(lstm_config_t lstmCfg);
int lstm_config_set_outputs(lstm_config_t lstmCfg, int outputs);
int lstm_config_get_outputs(lstm_config_t lstmCfg);
int lstm_config_set_hidden_layers(lstm_config_t lstmCfg, int hiddenLayers);
int lstm_config_get_hidden_layers(lstm_config_t lstmCfg);
int lstm_config_set_hidden_nodes(lstm_config_t lstmCfg, int hiddenLayerIndex, int hiddenNodes);
int lstm_config_get_hidden_nodes(lstm_config_t lstmCfg, int hiddenLayerIndex);
int lstm_config_set_input_transfer_func(lstm_config_t lstmCfg, int tFuncID);
int lstm_config_get_input_transfer_func(lstm_config_t lstmCfg);
int lstm_config_set_output_transfer_func(lstm_config_t lstmCfg, int tFuncID);
int lstm_config_get_output_transfer_func(lstm_config_t lstmCfg);
void lstm_config_set_learning_rate(lstm_config_t lstmCfg, float lRate);
float lstm_config_get_learning_rate(lstm_config_t lstmCfg);
void lstm_config_set_momentum_coef(lstm_config_t lstmCfg, float mCoef);
float lstm_config_get_momentum_coef(lstm_config_t lstmCfg);

int lstm_config_import(lstm_config_t* lstmCfgPtr, const char* filePath);
int lstm_config_export(lstm_config_t lstmCfg, const char* filePath);

int lstm_get_transfer_func_id(const char* tFuncName);

lstm_config_t lstm_get_config(lstm_t lstm);

/**
 * @}
 */

/**
 *	\~English @defgroup LSTM LSTM Functions
 *	@brief Functions of LSTM networks.
 *
 * @{
 *
 *	\~Chinese-Traditional @defgroup LSTM LSTM 函式
 *	@brief LSTM 網路函式
 *
 * @{
 */

int lstm_create(lstm_t* lstmPtr, lstm_config_t lstmCfg);
int lstm_clone(lstm_t* lstmPtr, lstm_t lstmSrc);
void lstm_delete(lstm_t lstm);

int lstm_import(lstm_t* lstmPtr, const char* filePath);
int lstm_export(lstm_t lstm, const char* filePath);

void lstm_rand_network(lstm_t lstm);
void lstm_zero_network(lstm_t lstm);

void lstm_forward_computation(lstm_t lstm, float* input, float* output);
void lstm_forward_computation_erase(lstm_t lstm);

int lstm_bptt_set_max_timestep(lstm_t lstm, int timeStep);
void lstm_bptt_sum_gradient(lstm_t lstm, float* dError);
void lstm_bptt_adjust_network(lstm_t lstm, float lRate, float mCoef, float gradLimit);
void lstm_bptt_erase(lstm_t lstm);

int lstm_training_gradient(lstm_t lstm, float** inputList, float** desireList, float** outputList, float** errList, int timeStep, float gradLimit);
int lstm_training_gradient_custom(lstm_t lstm, float lRate, float mCoef, float** inputList, float** desireList, float** outputList, float** errList, int timeStep, float gradLimit);

/**
 * @}
 */

/**
 *	\~English @defgroup State LSTM State Processing
 *	@brief Functions for LSTM state processing.
 *
 * @{
 *
 *	\~Chinese-Traditional @defgroup State LSTM 狀態處理
 *	@brief LSTM 狀態處理
 *
 * @{
 */

int lstm_state_create(lstm_state_t* lstmStatePtr, lstm_config_t lstmCfg);
void lstm_state_delete(lstm_state_t lstmState);
void lstm_state_erase(lstm_state_t lstmState);

int lstm_state_save(lstm_state_t lstmState, lstm_t lstm);
int lstm_state_restore(lstm_state_t lstmState, lstm_t lstm);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
