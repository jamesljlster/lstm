#ifndef _DEBUG_H_
#define _DEBUG_H_

#ifdef DEBUG
#include <stdio.h>
#define LOG(msg, ...) 							\
	fprintf(stderr, "%s(): ", __FUNCTION__);	\
	fprintf(stderr, msg, ##__VA_ARGS__);		\
	fprintf(stderr, "\n");
#else
#define	LOG(msg, ...)
#endif

#endif
