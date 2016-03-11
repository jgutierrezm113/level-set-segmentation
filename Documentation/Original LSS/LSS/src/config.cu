#include "config.h"

#if defined(CUDA_DEBUG)
__device__ double *debug_buff;
void initialize_debug_buff(size_t size)
{
	double *debug_buff_host;
	cudaMalloc(&debug_buff_host, sizeof(double) * size);
	cudaMemcpyToSymbol(debug_buff, &debug_buff_host, sizeof(double *));
}

void get_debug_buff(void **debug_buff_host, size_t size)
{
	*debug_buff_host = malloc(size);
	double *debug_buff_device;
	cudaMemcpyFromSymbol(&debug_buff_device, debug_buff, sizeof(double *));
	cudaMemcpy(*debug_buff_host, debug_buff_device, size, cudaMemcpyDeviceToHost);
}

__device__ void dump(void *data, size_t size)
{
	memcpy(debug_buff, data, size);
}


#endif
