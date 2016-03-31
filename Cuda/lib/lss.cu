/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Level Set Segmentation for Image Processing 
 *  
 */
 
#include "lss.h"

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

using namespace std;

void modMaxIter (int value){
	max_iterations = value;
}

/*
 * Lss Step 1 from Pseudo Code
 */
__global__ void lssStep1(unsigned int* intensity, 
			 unsigned int* labels,
			 signed int* phi, 
			 int targetLabel, 
			 int lowerIntensityBound, 
			 int upperIntensityBound,
			 int* globalBlockIndicator) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int blockId = by*gridDim.x+bx;
				
	// Including border
	__shared__ signed int intensityTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // input
	__shared__ signed int     labelTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // input
	__shared__ signed int       phiTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // output
	
	// Global Block Indicator
	__shared__ volatile signed int localGBI;
	__shared__ volatile signed int change;
		
	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	int x = bx*BLOCK_TILE_SIZE+tx;
	int y = by*BLOCK_TILE_SIZE+ty;
	  
	int location = 	y*(gridDim.x*BLOCK_TILE_SIZE)+x;
		
	int intensityData = intensity[location];
	int     labelData = labels[location];
	
	int sharedX = tx+1;
	int sharedY = ty+1;
		
	    labelTile[sharedY][sharedX] = labelData;
	intensityTile[sharedY][sharedX] = intensityData;

	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////
	int posX;
	int posY;
	
	// Horizontal Border
	if (ty == 0){
		posX = sharedX;
		posY = 0;
		if (by == 0){
			intensityTile[posY][posX] = 0;
			    labelTile[posY][posX] = 0;
		} else {
			intensityTile[posY][posX] = intensity[(y-1)*(gridDim.x*BLOCK_TILE_SIZE)+x];
			    labelTile[posY][posX] = labels[(y-1)*(gridDim.x*BLOCK_TILE_SIZE)+x];		
		}
	} else if (ty == BLOCK_TILE_SIZE-1){
		posX = sharedX;
		posY = BLOCK_TILE_SIZE+1;
		if (by == gridDim.y-1){
			intensityTile[posY][posX] = 0;
			    labelTile[posY][posX] = 0;
		} else {
			intensityTile[posY][posX] = intensity[(y+1)*(gridDim.x*BLOCK_TILE_SIZE)+x];
			    labelTile[posY][posX] = labels[(y+1)*(gridDim.x*BLOCK_TILE_SIZE)+x];
		}
		
	}
	if (ty == 0 || ty == BLOCK_TILE_SIZE-1){
		if (labelTile[posY][posX] == targetLabel && 
		    intensityTile[posY][posX] >= lowerIntensityBound && 
		    intensityTile[posY][posX] <= upperIntensityBound){
			phiTile[posY][posX] = 1;
		} else if (intensityTile[posY][posX] >= lowerIntensityBound && 
			   intensityTile[posY][posX] <= upperIntensityBound) {
			phiTile[posY][posX] = 2;
		} else {
			phiTile[posY][posX] = 0;
		}
	}
	
	// Vertical Border
	if (tx == 0){
		posX = 0;
		posY = sharedY;
		if (bx == 0){
			intensityTile[posY][posX] = 0;
			    labelTile[posY][posX] = 0;
		} else {
			intensityTile[posY][posX] = intensity[y*(gridDim.x*BLOCK_TILE_SIZE)+(x-1)];
			    labelTile[posY][posX] = labels[y*(gridDim.x*BLOCK_TILE_SIZE)+(x-1)];		
		}
	} else if (tx == BLOCK_TILE_SIZE-1){
		posX = BLOCK_TILE_SIZE+1;
		posY = sharedY;
		if (bx == gridDim.x-1){
			intensityTile[posY][posX] = 0;
			    labelTile[posY][posX] = 0;
		} else {
			intensityTile[posY][posX] = intensity[y*(gridDim.x*BLOCK_TILE_SIZE)+(x+1)];
			    labelTile[posY][posX] = labels[y*(gridDim.x*BLOCK_TILE_SIZE)+(x+1)];
		}
		
	}
	if (tx == 0 || tx == BLOCK_TILE_SIZE-1){
		if (labelTile[posY][posX] == targetLabel && 
		    intensityTile[posY][posX] >= lowerIntensityBound && 
		    intensityTile[posY][posX] <= upperIntensityBound){
			phiTile[posY][posX] = 1;
		} else if (intensityTile[posY][posX] >= lowerIntensityBound && 
			   intensityTile[posY][posX] <= upperIntensityBound) {
			phiTile[posY][posX] = 2;
		} else {
			phiTile[posY][posX] = 0;
		}
	}	
	localGBI = 0;
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////////////////////
	
	// Initialization
	if(intensityTile[sharedY][sharedX] >= lowerIntensityBound && 
	   intensityTile[sharedY][sharedX] <= upperIntensityBound) {
		localGBI = 1;
		if (labelTile[sharedY][sharedX] == targetLabel)
			phiTile[sharedY][sharedX] = 1;
		else
			phiTile[sharedY][sharedX] = 2;
	} else
		phiTile[sharedY][sharedX] = 0;
   
	// Work
	change = 1;
	
	__syncthreads();
	
	while (change){
		__syncthreads();
		change = 0;
		__syncthreads();
		
		if((phiTile[sharedY+1][sharedX]  == 1 ||
		    phiTile[sharedY-1][sharedX]  == 1 ||
		    phiTile[sharedY][sharedX+1]  == 1 ||
		    phiTile[sharedY][sharedX-1]  == 1 ) && 
		    phiTile[sharedY][sharedX]  == 2){
			phiTile  [sharedY][sharedX] = 1;
			change = 1;
		}
		__syncthreads();
	}
	
	// Write back to main memory
	phi[location] = phiTile[sharedY][sharedX];
	
	if (tx == 0 && ty == 0)
		globalBlockIndicator[blockId]=localGBI;

}

/*
 * Lss Step 2 from Pseudo Code
 */
 __global__ void lssStep2(signed int* phi, 
			 int* globalBlockIndicator,
			 int* globalFinishedVariable){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	int blockId = by*gridDim.x+bx;
	
	// Including border
	__shared__ signed int    phiTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // input/output

	// Global Block Indicator
	__shared__ volatile signed int borderChange;
	__shared__ volatile signed int change;
		
	// Read Global Block Indicator from global memory
	int localGBI = globalBlockIndicator[blockId];
	
	__syncthreads();
	
	if (localGBI) {
		// Read Input Data into Shared Memory
		/////////////////////////////////////////////////////////////////////////////

		int x = bx*BLOCK_TILE_SIZE+tx;
		int y = by*BLOCK_TILE_SIZE+ty;
		  
		int location = 	y*(gridDim.x*BLOCK_TILE_SIZE)+x;
			
		int sharedX = tx+1;
		int sharedY = ty+1;
			
		phiTile[sharedY][sharedX] = phi[location];

		// Read Global Finished Variable from main memory
		int threadGFV = *globalFinishedVariable;
		
		if (threadGFV > 0){

			// Read Border Data into Shared Memory
			/////////////////////////////////////////////////////////////////////////////////////
			int posX;
			int posY;
			
			// Horizontal Border
			if (ty == 0){
				posX = sharedX;
				posY = 0;
				if (by == 0){
					phiTile[posY][posX] = 0;
				} else {
					phiTile[posY][posX] = phi[(y-1)*(gridDim.x*BLOCK_TILE_SIZE)+x];		
				}
			} else if (ty == BLOCK_TILE_SIZE-1){
				posX = sharedX;
				posY = BLOCK_TILE_SIZE+1;
				if (by == gridDim.y-1){
					phiTile[posY][posX] = 0;
				} else {
					phiTile[posY][posX] = phi[(y+1)*(gridDim.x*BLOCK_TILE_SIZE)+x];
				}
			}
			
			// Vertical Border
			if (tx == 0){
				posX = 0;
				posY = sharedY;
				if (bx == 0){
					phiTile[posY][posX] = 0;
				} else {
					phiTile[posY][posX] = phi[y*(gridDim.x*BLOCK_TILE_SIZE)+(x-1)];		
				}
			} else if (tx == BLOCK_TILE_SIZE-1){
				posX = BLOCK_TILE_SIZE+1;
				posY = sharedY;
				if (bx == gridDim.x-1){
					phiTile[posY][posX] = 0;
				} else {
					phiTile[posY][posX] = phi[y*(gridDim.x*BLOCK_TILE_SIZE)+(x+1)];
				}
			}
			
			borderChange = 0; // Shared variable
			change = 1; // Shared variable
			__syncthreads();
			
			// Algorithm 
			/////////////////////////////////////////////////////////////////////

			while (change){
				__syncthreads();
				change = 0;
				__syncthreads();
				
				if((phiTile[sharedY+1][sharedX]  == 1 ||
				    phiTile[sharedY-1][sharedX]  == 1 ||
				    phiTile[sharedY][sharedX+1]  == 1 ||
				    phiTile[sharedY][sharedX-1]  == 1 ) && 
				    phiTile[sharedY][sharedX]  == 2){
					phiTile[sharedY][sharedX] = 1;
					change = 1;
					if (sharedX == 1 || sharedX == BLOCK_TILE_SIZE || 
					    sharedY == 1 || sharedY == BLOCK_TILE_SIZE ){
						borderChange = 1;
					}
				}
				__syncthreads();
			}
			
			if (borderChange){
				phi[location] = phiTile[sharedY][sharedX];
			} else {
				if (tx == 0 && ty == 0 ){
					atomicSub (globalFinishedVariable,1);
				}
			}
		} // GFV > 0 indicating other blocks still are doing work
	} else { // Local GBI = 1, means block has to do work
		if (tx == 0 && ty == 0){
			atomicSub (globalFinishedVariable,1);
		}
	}
}

/*
 * Lss Step 3 from Pseudo Code
 */
__global__ void lssStep3(signed int* phi, 
			 int* globalBlockIndicator) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
				
	// Including border
	__shared__ signed int    phiTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // input
	__shared__ signed int phiOutTile[BLOCK_TILE_SIZE+2][BLOCK_TILE_SIZE+2]; // output

	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	int x = bx*BLOCK_TILE_SIZE+tx;
	int y = by*BLOCK_TILE_SIZE+ty;
	  
	int location = 	y*(gridDim.x*BLOCK_TILE_SIZE)+x;
		
	int sharedX = tx+1;
	int sharedY = ty+1;
		
	phiTile[sharedY][sharedX] = phi[location];

	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////
	int posX;
	int posY;
	
	// Horizontal Border
	if (ty == 0){
		posX = sharedX;
		posY = 0;
		if (by == 0){
			phiTile[posY][posX] = 0;
		} else {
			phiTile[posY][posX] = phi[(y-1)*(gridDim.x*BLOCK_TILE_SIZE)+x];		
		}
	} else if (ty == BLOCK_TILE_SIZE-1){
		posX = sharedX;
		posY = BLOCK_TILE_SIZE+1;
		if (by == gridDim.y-1){
			phiTile[posY][posX] = 0;
		} else {
			phiTile[posY][posX] = phi[(y+1)*(gridDim.x*BLOCK_TILE_SIZE)+x];
		}
	}
	
	// Vertical Border
	if (tx == 0){
		posX = 0;
		posY = sharedY;
		if (bx == 0){
			phiTile[posY][posX] = 0;
		} else {
			phiTile[posY][posX] = phi[y*(gridDim.x*BLOCK_TILE_SIZE)+(x-1)];		
		}
	} else if (tx == BLOCK_TILE_SIZE-1){
		posX = BLOCK_TILE_SIZE+1;
		posY = sharedY;
		if (bx == gridDim.x-1){
			phiTile[posY][posX] = 0;
		} else {
			phiTile[posY][posX] = phi[y*(gridDim.x*BLOCK_TILE_SIZE)+(x+1)];
		}
	}
		
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////////////////////

	if(phiTile[sharedY][sharedX] == 1) {
		if(phiTile[sharedY+1][sharedX]  == 1 &&
		   phiTile[sharedY-1][sharedX]  == 1 &&
		   phiTile[sharedY][sharedX+1]  == 1 &&
		   phiTile[sharedY][sharedX-1]  == 1 ){
			phiOutTile[sharedY][sharedX] = 0xFD;
		} else 
			phiOutTile[sharedY][sharedX] = 0xFF;
	} else
		if(phiTile[sharedY+1][sharedX]  == 1 ||
		   phiTile[sharedY-1][sharedX]  == 1 ||
		   phiTile[sharedY][sharedX+1]  == 1 ||
		   phiTile[sharedY][sharedX-1]  == 1 ){
			phiOutTile[sharedY][sharedX] = 1;
		} else 
			phiOutTile[sharedY][sharedX] = 3;
	
	// Write back to main memory
	phi[location] = phiOutTile[sharedY][sharedX];
}

__global__ void evolveContour(unsigned int* intensity, 
			      unsigned int* labels,
			      signed int* phi, 
			      int gridXSize,
			      int gridYSize,
			      int* targetLabels, 
			      int* lowerIntensityBounds, 
			      int* upperIntensityBounds,
			      int max_iterations, 
			      int* globalBlockIndicator,
			      int* globalFinishedVariable ) {
        int tid = threadIdx.x;
	
	// Size in ints
	int size = (gridXSize*BLOCK_TILE_SIZE)*(gridYSize*BLOCK_TILE_SIZE);
	
	// New phi pointer for each label.
	phi   = &phi[tid*size];

	globalBlockIndicator = &globalBlockIndicator[tid*gridXSize*gridYSize];
	*globalBlockIndicator = 0;

	// Global synchronization variable
	globalFinishedVariable = &globalFinishedVariable[tid];
       
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Initialize phi array
	lssStep1<<<dimGrid, dimBlock>>>(intensity, 
					labels,  
					phi, 
					targetLabels[tid], 
					lowerIntensityBounds[tid], 
					upperIntensityBounds[tid],
					globalBlockIndicator );
	int iterations;
	do {
		iterations++;
		*globalFinishedVariable = gridXSize*gridYSize;
		lssStep2<<<dimGrid, dimBlock>>>(phi, 
					globalBlockIndicator,
					globalFinishedVariable );
	} while ((*globalFinishedVariable > 0) && (iterations < max_iterations));
	
	lssStep3<<<dimGrid, dimBlock>>>(phi, 
					globalBlockIndicator );
	
}

signed int *levelSetSegment(unsigned int *intensity, 
			    unsigned int *labels,
			    int height, 
			    int width,
			    int *targetLabels, 
			    int *lowerIntensityBounds,
			    int *upperIntensityBounds,
			    int numLabels){
	
	#if defined(VERBOSE)
		printf("Printing input data\n");
		printf("Height: %d\n", height);
		printf("Width: %d\n", width);
		printf("Num Labels: %d\n", numLabels);
		
		for (int i = 0; i < numLabels; i++){
			printf("target label: %d\n", targetLabels[i]);
			printf("lower bound: %d\n", lowerIntensityBounds[i]);
			printf("upper bound: %d\n", upperIntensityBounds[i]);	
		}
	#endif
	
	int gridXSize = 1 + (( width - 1) / BLOCK_TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / BLOCK_TILE_SIZE);
	
	int XSize = gridXSize*BLOCK_TILE_SIZE;
	int YSize = gridYSize*BLOCK_TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	#if defined(VERBOSE)
		printf ("Allocating arrays in GPU memory.\n");
	#endif
	
	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif
	
	checkCuda(cudaMalloc((void**)&gpu.targetLabels           , numLabels*sizeof(int)));
        checkCuda(cudaMalloc((void**)&gpu.lowerIntensityBounds   , numLabels*sizeof(int)));
        checkCuda(cudaMalloc((void**)&gpu.upperIntensityBounds   , numLabels*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.labels                 , gpu.size*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.phi                    , numLabels*gpu.size*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.globalBlockIndicator   , numLabels*gridXSize*gridYSize*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.globalFinishedVariable , numLabels*sizeof(int)));
	
	// Allocate result array in CPU memory
	gpu.phiOnCpu = new signed int[gpu.size*numLabels];
	
        checkCuda(cudaMemcpy(gpu.targetLabels, 
			targetLabels, 
			numLabels*sizeof(int), 
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(gpu.lowerIntensityBounds, 
			lowerIntensityBounds, 
			numLabels*sizeof(int), 
			cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(gpu.upperIntensityBounds, 
			upperIntensityBounds, 
			numLabels*sizeof(int), 
			cudaMemcpyHostToDevice));
			
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(int), 
			cudaMemcpyHostToDevice));
			
        checkCuda(cudaMemcpy(gpu.labels, 
			labels, 
			gpu.size*sizeof(int), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
	
	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif
	
	// Launch kernel to begin image segmenation
	evolveContour<<<1, numLabels>>>(gpu.intensity, 
					gpu.labels,
					gpu.phi, 
					gridXSize,
					gridYSize, 
					gpu.targetLabels, 
					gpu.lowerIntensityBounds, 
					gpu.upperIntensityBounds,
					max_iterations,
					gpu.globalBlockIndicator,
					gpu.globalFinishedVariable );
	
	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.phiOnCpu, 
			gpu.phi, 
			numLabels*gpu.size*sizeof(int), 
			cudaMemcpyDeviceToHost));
	
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.labels));
	checkCuda(cudaFree(gpu.phi));
	checkCuda(cudaFree(gpu.targetLabels));
	checkCuda(cudaFree(gpu.lowerIntensityBounds));
	checkCuda(cudaFree(gpu.upperIntensityBounds));
	checkCuda(cudaFree(gpu.globalBlockIndicator));
	checkCuda(cudaFree(gpu.globalFinishedVariable));
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif
	
	return(gpu.phiOnCpu);

}
