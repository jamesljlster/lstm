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

/** Type definition of lstm. @since 0.2.0 */
typedef struct LSTM_STRUCT* lstm_t;

/** Type definition of lstm configuration. @since 0.2.0 */
typedef struct LSTM_CONFIG_STRUCT* lstm_config_t;

/**
 *@}
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 *	\~English @defgroup Config Configuration
 *	@brief Configuration of neural netwrok.
 *
 *@{
 *
 *	\~Chinese-Traditional @defgroup Config 配置
 *	@brief 類神經網路配置
 *
 *@{
 */

/**
 *@}
 */

#ifdef __cplusplus
}
#endif

#endif
