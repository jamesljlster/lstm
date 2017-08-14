/**
 *	@author	Zheng-Ling Lai <jamesljlster@gmail.com>
 *	@file	lstm.h
**/

#ifndef __LSTM_H__
#define __LSTM_H__

/**
 *	\~English @defgroup Enum Enumeration
 *	@brief Enumeration of the Library.
 *
 *@{
 *
 *	\~Chinese-Traditional @defgroup Enum 列舉
 *	@brief 函式庫列舉
 *
 *@{
 */

/** 
 *	\~English 
 *		Return value definitions of LSTM library.
 *
 *	\~Chinese-Traditional
 *		LSTM 函式庫回傳值定義
 */
enum LSTM_RETUEN_VALUE
{
	LSTM_NO_ERROR		= 0,	/*!< No error occured while running called function. @since 0.1.0 */
	LSTM_MEM_FAILED		= -1,	/*!< Memory allocation failed. @since 0.1.0 */
};

/** Transfer (activation) function index definitions. */
enum LSTM_TRANSFER_FUNC
{
	LSTM_SIGMOID				= 0,	/*!< Sigmoid function. @since 0.1.0 */
	LSTM_MODIFIED_SIGMOID	= 1,	/*!< Modified sigmoid function. @since 0.1.0 */
	LSTM_HYPERBOLIC_TANGENT	= 2,	/*!< Hyperbolic tangent function. @since 0.1.0 */
	LSTM_GAUSSIAN			= 3,	/*!< Gaussian function. @since 0.1.0 */
	LSTM_BENT_IDENTITY		= 4,	/*!< Bent identity function. @since 0.1.0 */
	LSTM_SOFTPLUS			= 5,	/*!< SoftPlus function. @since 0.1.0 */
	LSTM_SOFTSIGN			= 6,	/*!< SoftSign function. @since 0.1.0 */
	LSTM_SINC				= 7,	/*!< Sinc function. @since 0.1.0 */
	LSTM_SINUSOID			= 8,	/*!< Sinusoid (sine) function. @since 0.1.0 */
	LSTM_IDENTITY			= 9,	/*!< Identity function. @since 0.1.0 */
	LSTM_RELU				= 10,	/*!< Rectifier linear unit function. @since 0.1.0 */
};

/**
 *@}
 */

/**
 *	\~English @defgroup Types Data Types
 *	@brief Data Types of the Library.
 *
 *@{
 *
 *	\~Chinese-Traditional @defgroup Types 資料型別
 *	@brief 函式庫資料型別
 *
 *@{
 */

/** Type definition of lstm. @since 0.1.0 */
typedef struct LSTM_STRUCT* lstm_t;

/** Type definition of lstm configuration. @since 0.1.0 */
typedef struct LSTM_CONFIG_STRUCT* lstm_config_t;

/**
 *@}
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 *	\~English @defgroup Config Configuration
 *	@brief Configuration of LSTM networks.
 *
 *@{
 *
 *	\~Chinese-Traditional @defgroup Config 配置
 *	@brief LSTM 網路配置
 *
 *@{
 */

int lstm_config_create(lstm_config_t* lstmCfgPtr);
void lstm_config_delete(lstm_config_t lstmCfg);

/**
 *@}
 */

#ifdef __cplusplus
}
#endif

#endif
