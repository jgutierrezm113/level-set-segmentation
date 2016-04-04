#include "lss.h"
#include "utility.h"

#include <cstdio>

__device__ udat *difference;
__device__ int *s1, *s2, *cnt;
__device__ float c1, c2;
__device__ volatile int condition;

// Initialization -------------------------------------------------------------------------------
__global__ void initDifference(udat *differenceDev, udat *intensityDev, udat* componentDev, int size, int targetLabel)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < size)
	{
		if (componentDev[id] == targetLabel)
		{
			differenceDev[id] = intensityDev[id];
		}
		else
		{
			differenceDev[id] = 0;
		}
	}
}

__global__ void initPhi(udat* componentDev, dat* phiDev, int height, int width, int targetLabel)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xpos = 30 * bx + tx;
	int ypos = 30 * by + ty;

	int phi;
	__shared__ int tile[32][32];

	// Load data into shared memory and registers
	if(xpos < width && ypos < height)
	{
		tile[ty][tx] = componentDev[ypos * width + xpos];
	}

	// Initialization
	if(tx > 0 && tx < 31 && ty > 0 && ty < 31 && xpos < width - 1 && ypos < height - 1)
	{
		// Phi
		if(tile[ty][tx] != targetLabel)
		{
			if(tile[ty][tx - 1] != targetLabel && tile[ty][tx + 1] != targetLabel && tile[ty - 1][tx] != targetLabel && tile[ty + 1][tx] != targetLabel)
				phi = 3;
			else
				phi = 1;
		}
		else
		{
			if(tile[ty][tx - 1] != targetLabel || tile[ty][tx + 1] != targetLabel || tile[ty - 1][tx] != targetLabel || tile[ty + 1][tx] != targetLabel)
				phi = -1;
			else
				phi = -3;
		}

		// Load data back into global memory
		phiDev[ypos * width + xpos] = phi;
	}
}

__global__ void sumImage(int *s1Dev, int *s2Dev, int *cntDev, udat *differenceDev, dat *phiDev, int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	__shared__ int inside[512];
	__shared__ int outside[512];
	__shared__ int counter[512];

	if (id < size)
	{
		int difference = differenceDev[id];
		if (phiDev[id] > 0)
		{
			inside[tid] = difference;
			outside[tid] = 0;
			counter[tid] = 1;
		}
		else
		{
			inside[tid] = 0;
			outside[tid] = difference;
			counter[tid] = 0;
		}
	}
	else
	{
		inside[tid] = 0;
		outside[tid] = 0;
		counter[tid] = 0;
	}
	__syncthreads();

	if (tid < 256)
	{
		inside[tid] += inside[tid + 256];
		outside[tid] += outside[tid + 256];
		counter[tid] += counter[tid + 256];
	}
	__syncthreads();

	if (tid < 128)
	{
		inside[tid] += inside[tid + 128];
		outside[tid] += outside[tid + 128];
		counter[tid] += counter[tid + 128];
	}
	__syncthreads();

	if (tid < 64)
	{
		inside[tid] += inside[tid + 64];
		outside[tid] += outside[tid + 64];
		counter[tid] += counter[tid + 64];
	}
	__syncthreads();

	if (tid < 32)
	{
		inside[tid] += inside[tid + 32];
		inside[tid] += inside[tid + 16];
		inside[tid] += inside[tid + 8];
		inside[tid] += inside[tid + 4];
		inside[tid] += inside[tid + 2];
		inside[tid] += inside[tid + 1];

		outside[tid] += outside[tid + 32];
		outside[tid] += outside[tid + 16];
		outside[tid] += outside[tid + 8];
		outside[tid] += outside[tid + 4];
		outside[tid] += outside[tid + 2];
		outside[tid] += outside[tid + 1];

		counter[tid] += counter[tid + 32];
		counter[tid] += counter[tid + 16];
		counter[tid] += counter[tid + 8];
		counter[tid] += counter[tid + 4];
		counter[tid] += counter[tid + 2];
		counter[tid] += counter[tid + 1];
	}

	if(tid == 0)
	{
		s1Dev[blockIdx.x] = inside[0];
		s2Dev[blockIdx.x] = outside[0];
		cntDev[blockIdx.x] = counter[0];
	}
}

__global__ void averageImage(float *c1Dev, float *c2Dev, int *s1Dev, int *s2Dev, int *cntDev, int n, int size)
{
	unsigned int id = threadIdx.x;

	__shared__ int s1[512];
	__shared__ int s2[512];
	__shared__ int cnt[512];

	if (id < n)
	{
		s1[id] = s1Dev[id];
		s2[id] = s2Dev[id];
		cnt[id] = cntDev[id];
	}
	else
	{
		s1[id] = 0;
		s2[id] = 0;
		cnt[id] = 0;
	}
	__syncthreads();

	if (id < 256)
	{
		s1[id] += s1[id + 256];
		s2[id] += s2[id + 256];
		cnt[id] += cnt[id + 256];
	}
	__syncthreads();

	if (id < 128)
	{
		s1[id] += s1[id + 128];
		s2[id] += s2[id + 128];
		cnt[id] += cnt[id + 128];
	}
	__syncthreads();

	if (id < 64)
	{
		s1[id] += s1[id + 64];
		s2[id] += s2[id + 64];
		cnt[id] += cnt[id + 64];
	}
	__syncthreads();

	if (id < 32)
	{
		s1[id] += s1[id + 32];
		s1[id] += s1[id + 16];
		s1[id] += s1[id + 8];
		s1[id] += s1[id + 4];
		s1[id] += s1[id + 2];
		s1[id] += s1[id + 1];

		s2[id] += s2[id + 32];
		s2[id] += s2[id + 16];
		s2[id] += s2[id + 8];
		s2[id] += s2[id + 4];
		s2[id] += s2[id + 2];
		s2[id] += s2[id + 1];

		cnt[id] += cnt[id + 32];
		cnt[id] += cnt[id + 16];
		cnt[id] += cnt[id + 8];
		cnt[id] += cnt[id + 4];
		cnt[id] += cnt[id + 2];
		cnt[id] += cnt[id + 1];
	}

	if (id == 0)
	{
		*c1Dev = (float)s1[0] / cnt[0];
		*c2Dev = (float)s2[0] / (size - cnt[0]);
	}

}

__global__ void evolveContour(udat *intensityDev, udat *componentDev, dat *phiDev, int height, int width, int *labelsDev, int numberOfImages, int numberOfLabels)
//__global__ void evolveContour(udat *intensityDev, udat *componentDev, dat *phiDev, int height, int width, int imageId, int *labelsDev, int numberOfLabels)
{
	//int id = threadIdx.x;	//now = imageId, used to be 'labelId'
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id > numberOfImages * numberOfLabels)
		return;

	int imageId = id / numberOfLabels;
	int labelId = id % numberOfLabels;
	int size = height * width;
	int threadNumber = 128;
	int blockNumber = (size + threadNumber - 1) / threadNumber;
	dim3 dimBlock(32, 32);
	dim3 dimGrid(width / 30 + 1, height / 30 + 1);

	intensityDev = &intensityDev[imageId * size];
	componentDev = &componentDev[imageId * size];
	phiDev = &phiDev[(imageId * numberOfLabels + labelId) * size];
	//phiDev = &phiDev[(imageId * numberOfLabels + id) * size];

#if defined(CUDA_DEBUG)
	cudaDeviceSynchronize();
	printf("image id: %d, component address: %p, intensity address: %p", imageId, componentDev, intensityDev);
	dump(componentDev, sizeof(udat) * height * width);
#endif

	initDifference<<<blockNumber, threadNumber>>>(difference, intensityDev, componentDev, size, labelsDev[id]);

	initPhi<<<dimGrid, dimBlock>>>(componentDev, phiDev, height, width, labelsDev[id]);

	int numbefOfIterations = 0;
	condition = 1;
	while(condition)
	{
		condition = 0;
		numbefOfIterations++;

		sumImage<<<blockNumber, threadNumber>>>(s1, s2, cnt, difference, phiDev, size);
		averageImage<<<1, blockNumber>>>(&c1, &c2, s1, s2, cnt, blockNumber, size);

		// Outward evolution
		switchIn<<<dimGrid, dimBlock>>>(difference, c1, c2, phiDev, height, width);

		// Inward evolution
		switchOut<<<dimGrid, dimBlock>>>(difference, c1, c2, phiDev, height, width);

		// Check stopping condition
		printf("iteration %d\n", numbefOfIterations);
		if(numbefOfIterations % 3 == 0)
		{
			checkStop<<<blockNumber, threadNumber>>>(difference, c1, c2, phiDev, size);
		}
		else
			condition = 1;

		if(condition == 0 && numbefOfIterations % 3 == 0)
			cudaDeviceSynchronize();
	}
}

__global__ void switchIn(udat *differenceDev, float c1Dev, float c2Dev, signed char* phiDev, int height, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = 30 * bx + tx;
	int y = 30 * by + ty;

	__shared__ int phi[32][32];

	// Load data into shared memory and registers
	if(x < width && y < height)
	{
		phi[ty][tx] = phiDev[y * width + x];
	}

	// Outward evolve
	if(tx > 0 && tx < 31 && ty > 0 && ty < 31 && x < width - 1 && y < height - 1)
	{
		// Delete points from Lout and add them to Lin
		if (phi[ty][tx] == 1)
		{
			float diff = differenceDev[y * width + x];
			float speed = -((diff - c1Dev) * (diff - c1Dev)) + (diff - c2Dev) * (diff - c2Dev);
			if (speed > 0)
				phi[ty][tx] = -1;
		}

		// Update neighborhood
		if (phi[ty][tx] == 3)
		{
			if(phi[ty][tx - 1] == -1 || phi[ty][ tx + 1] == -1 || phi[ty - 1][tx] == -1 || phi[ty + 1][tx] == -1)
				phi[ty][tx] = 1;
		}

		// Eliminate redundant points in Lin
		if (phi[ty][tx] == -1)
		{
			if(phi[ty][tx - 1] < 0 && phi[ty][tx + 1] < 0 && phi[ty - 1][tx] < 0 && phi[ty + 1][tx] < 0)
				phi[ty][tx] = -3;
		}

		// Load data back into global memory
		phiDev[y * width + x] = phi[ty][tx];
	}
}

__global__ void switchOut(udat *differenceDev, float c1Dev, float c2Dev, signed char* phiDev, int height, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = 30 * bx + tx;
	int y = 30 * by + ty;

	__shared__ int phi[32][32];

	// Load data into shared memory and registers
	if (x < width && y < height)
	{
		phi[ty][tx] = phiDev[y * width + x];
	}

	if (tx > 0 && tx < 31 && ty > 0 && ty < 31 && x < width - 1 && y < height - 1)
	{
		// Delete points from Lin and add them to Lout
		if (phi[ty][tx] == -1)
		{
			float diff = differenceDev[y * width + x];
			float speed = -((diff - c1Dev) * (diff - c1Dev)) + (diff - c2Dev) * (diff - c2Dev);
			if (speed < 0)
				phi[ty][tx] = 1;
		}

		// Update neighborhood
		if (phi[ty][tx] == -3)
		{
			if(phi[ty][tx - 1] == 1 || phi[ty][tx + 1] == 1 || phi[ty - 1][tx] == 1 || phi[ty + 1][tx] == 1)
				phi[ty][tx] = -1;
		}

		// Eliminate redundant points
		if (phi[ty][tx] == 1)
		{
			if(phi[ty][tx - 1] > 0 && phi[ty][tx + 1] > 0 && phi[ty - 1][tx] > 0 && phi[ty + 1][tx] > 0)
				phi[ty][tx] = 3;
		}

		// Load data back into global memory
		phiDev[y * width + x] = phi[ty][tx];
	}
}

__global__ void checkStop(udat *differenceDev, float c1Dev, float c2Dev, dat* phiDev, int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < size)
	{
		int phi = phiDev[id];

		if (phi == 1 || phi == -1)
		{
			float diff = differenceDev[id];
			float speed = -((diff - c1Dev) * (diff - c1Dev)) + (diff - c2Dev) * (diff - c2Dev);

			if((phi == 1 && speed > 0) || (phi == -1 && speed < 0))
				condition++;
		}
	}
}

signed char *levelSetSegment(unsigned char *intensity, unsigned char *component, int height, int width, int numberOfImages, int *labels, int numberOfLabels)
{
#if defined(CUDA_TIMING)
	EventTimer t1, t2;
	t1.start();
#endif

	udat *intensityDev, *componentDev;
	int *labelsDev;
	cudaMalloc(&intensityDev, sizeof(udat) * height * width * numberOfImages);
	cudaMalloc(&componentDev, sizeof(udat) * height * width * numberOfImages);
	cudaMalloc(&labelsDev, sizeof(int) * numberOfLabels);
#if defined(USE_INT)
	udat *intensiyTmp = new udat[height * width * numberOfImages];
	udat *componentTmp = new udat[height * width * numberOfImages];
	for (int i = 0; i < height * width * numberOfImages)
	{
		intensityTmp[i] = intensity[i];
		componentTmp[i] = component[i];
	}
	cudaMemcpy(intensityDev, intensityTmp, sizeof(udat) * height * width * numberOfImages, cudaMemcpyHostToDevice);
	cudaMemcpy(componentDev, componentTmp, sizeof(udat) * height * width * numberOfImages, cudaMemcpyHostToDevice);
#else
	cudaMemcpy(intensityDev, intensity, sizeof(udat) * height * width * numberOfImages, cudaMemcpyHostToDevice);
	cudaMemcpy(componentDev, component, sizeof(udat) * height * width * numberOfImages, cudaMemcpyHostToDevice);
#endif
	cudaMemcpy(labelsDev, labels, sizeof(int) * numberOfLabels, cudaMemcpyHostToDevice);

	dat *phiDev;
	cudaMalloc(&phiDev, sizeof(dat) * height * width * numberOfLabels * numberOfImages);

	udat *differenceDev;
	int *s1Dev, *s2Dev, *cntDev;
	cudaMalloc(&differenceDev, sizeof(udat) * height * width * numberOfLabels * numberOfImages);
	cudaMalloc(&s1Dev, sizeof(int) * 512 * numberOfLabels * numberOfImages);
	cudaMalloc(&s2Dev, sizeof(int) * 512 * numberOfLabels * numberOfImages);
	cudaMalloc(&cntDev, sizeof(int) * 512 * numberOfLabels * numberOfImages);
	cudaMemcpyToSymbol(difference, &differenceDev, sizeof(udat *));
	cudaMemcpyToSymbol(s1, &s1Dev, sizeof(int *));
	cudaMemcpyToSymbol(s2, &s2Dev, sizeof(int *));
	cudaMemcpyToSymbol(cnt, &cntDev, sizeof(int *));

	/*
	cudaStream_t stream[numberOfImages];
	for(int i=0; i<numberOfImages; i++)
		cudaStreamCreate(&stream[i]);
	*/

#if defined(CUDA_DEBUG)
	initialize_debug_buff(height * width);
	udat *intensity_debug = new udat[height * width];
	udat *component_debug = new udat[height * width];
	cudaMemcpy(intensity_debug, intensityDev, sizeof(udat) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(component_debug, componentDev, sizeof(udat) * height * width, cudaMemcpyDeviceToHost);
	saveImage(intensity_debug, height, width, "intensity.ppm");
	saveImage(component_debug, height, width, "component.ppm");
#endif

#if defined(CUDA_TIMING)
	t2.start();
#endif

	int maxThreads = 1024;
	int numImagesPerBlock = (maxThreads / numberOfLabels);
	int numThreadsPerBlock = numImagesPerBlock * numberOfLabels;
	int numBlocks = ceil((float)numberOfImages / (float)numImagesPerBlock);

	printf("numBlocks %d\n", numBlocks);


	//for(int i = 0; i < numberOfImages; i++)
		evolveContour<<<numBlocks, numThreadsPerBlock, 0>>>(intensityDev, componentDev, phiDev, height, width, labelsDev, numberOfImages, numberOfLabels);
		//evolveContour<<<numberOfImages, numberOfLabels, 0>>>(intensityDev, componentDev, phiDev, height, width, labelsDev, numberOfImages, numberOfLabels);
		//evolveContour<<<1, numberOfLabels, 0, stream[i]>>>(intensityDev, componentDev, phiDev, height, width, i, labelsDev, numberOfLabels);

#if defined(CUDA_DEBUG)
	udat *output_debug;
	get_debug_buff((void **)&output_debug, height * width * sizeof(udat));
	saveImage((unsigned char *)output_debug, height, width);
#endif

#if defined(CUDA_TIMING)
	cudaDeviceSynchronize();
	t2.stop();
#endif

#if defined(USE_INT)
	dat *phiTmp = new dat[height * width * numberOfLabels * numberOfImages];
	cudaMemcpy(phiTmp, phiDev, sizeof(dat) * height * width * numberOfLabels * numberOfImages, cudaMemcpyDeviceToHost);
	unsigned char *phi = new unsigned char[height * width * numberOfLabels * numberOfImages];
	for (int i = 0; i < height * width * numberOfLabels * numberOfImages)
	{
		phi[i] = phiTmp[i];
	}
#else
	dat *phi = new dat[height * width * numberOfLabels * numberOfImages];
	cudaMemcpy(phi, phiDev, sizeof(dat) * height * width * numberOfLabels * numberOfImages, cudaMemcpyDeviceToHost);
#endif

	cudaFree(cntDev);
	cudaFree(s2Dev);
	cudaFree(s1Dev);
	cudaFree(differenceDev);
	cudaFree(phiDev);
	cudaFree(labelsDev);
	cudaFree(componentDev);
	cudaFree(intensityDev);

#if defined(CUDA_TIMING)
	t1.stop();
	printf("Computation time %fs\n", t2.elapsed());
	printf("Total time %fs\n", t1.elapsed());
#endif
	return phi;
}
