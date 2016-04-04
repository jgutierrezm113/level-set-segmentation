#ifndef CONFIG_H
#define CONFIG_H

#include <cstdlib>

// data type
/*#define USE_INT*/

#if defined(USE_INT)
	typedef unsigned int udat;
	typedef signed int dat;
#else
	typedef unsigned char udat;
	typedef signed char dat;
#endif


// debug config directives
/*#define CUDA_DEBUG*/
#define ENABLE_DUMPER

#if defined(CUDA_DEBUG)
extern __device__ double *debug_buff;
extern void initialize_debug_buff(size_t size);
extern void get_debug_buff(void **debug_buff_host, size_t size);
extern __device__ void dump(void *data, size_t size);
#endif


// timing directives
#define CUDA_TIMING

#if defined(CUDA_TIMING)
#include "eventTimer.h"
#endif

#endif

