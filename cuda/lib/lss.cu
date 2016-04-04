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
			 int* globalBlockIndicator,
			 int* globalFinishedVariable) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int blockId = by*gridDim.x+bx;
				
	// Including border
	__shared__ unsigned char intensityTile[TILE_SIZE+2][TILE_SIZE+2]; // input
	__shared__ unsigned char     labelTile[TILE_SIZE+2][TILE_SIZE+2]; // input
	__shared__   signed char       phiTile[TILE_SIZE+2][TILE_SIZE+2]; // output
	
	// Flags
	__shared__ volatile int localGBI;
		
	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	int x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	int y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	  
	int location = 	(((x>>TTSB)&BTSMask)                ) |
			(((y>>TTSB)&BTSMask) << BTSB        ) |
			((x>>TSB)            << (BTSB+BTSB) ) ;
	location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
		
	int intensityData = intensity[location];
	int     labelData = labels[location];
	
	int sharedX = tx*THREAD_TILE_SIZE+1;
	int sharedY = ty*THREAD_TILE_SIZE+1;
		
	labelTile[sharedY  ][sharedX  ] = labelData         & 0xFF;
	labelTile[sharedY  ][sharedX+1] = (labelData >>  8) & 0xFF;
	labelTile[sharedY+1][sharedX  ] = (labelData >> 16) & 0xFF;
	labelTile[sharedY+1][sharedX+1] = (labelData >> 24) & 0xFF;
	
	intensityTile[sharedY  ][sharedX  ] = intensityData         & 0xFF;
	intensityTile[sharedY  ][sharedX+1] = (intensityData >>  8) & 0xFF;
	intensityTile[sharedY+1][sharedX  ] = (intensityData >> 16) & 0xFF;
	intensityTile[sharedY+1][sharedX+1] = (intensityData >> 24) & 0xFF;

	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	// Registers meant for speed. Two given each thread will update 2 pixels.
	int shiftTileReg1 = 0;
	int shiftTileReg2 = 0;
	
	int borderXLoc = 0;
	int borderYLoc = 0;
	
	// Needed Variables
	int bLocation;
	int borderIntData;
	int borderLabData;
	
	// Update horizontal border
	borderXLoc = (tx<<TTSB)+1;
	if (ty == 0 ){		
		// Location to write in shared memory
		borderYLoc = 0;
		if (by != 0) {
			// Upper block border
			y-=THREAD_TILE_SIZE;
			shiftTileReg1 = 16;
			shiftTileReg2 = 24;
		}
	} else if (ty == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderYLoc = TILE_SIZE+1;			
		if (by != gridDim.y-1) {
			// Lower block border
			y+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 8;
		}
	}
	// Read from global and write to shared memory
	if (ty == 0 || ty == BLOCK_TILE_SIZE-1) {
		if ((by == 0           && ty == 0                ) || 
		    (by == gridDim.y-1 && ty == BLOCK_TILE_SIZE-1)){
			intensityTile[borderYLoc][borderXLoc  ] = 0;
			intensityTile[borderYLoc][borderXLoc+1] = 0;
			    labelTile[borderYLoc][borderXLoc  ] = 0;
			    labelTile[borderYLoc][borderXLoc+1] = 0;
			      phiTile[borderYLoc][borderXLoc  ] = 0;
			      phiTile[borderYLoc][borderXLoc+1] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                ) |
				    (((y>>TTSB)&BTSMask) << BTSB        ) |
				     ((x>>TSB)            << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderIntData = intensity[bLocation];
			borderLabData =    labels[bLocation];
			
			intensityTile[borderYLoc][borderXLoc  ] 
					= ( borderIntData >> shiftTileReg1 ) & 0xFF;
			intensityTile[borderYLoc][borderXLoc+1] 
					= ( borderIntData >> shiftTileReg2 ) & 0xFF;
			    labelTile[borderYLoc][borderXLoc  ] 
					= ( borderLabData >> shiftTileReg1 ) & 0xFF;
			    labelTile[borderYLoc][borderXLoc+1] 
					= ( borderLabData >> shiftTileReg2 ) & 0xFF;
			// Update Phi of border
			if (ty == 0 || ty == BLOCK_TILE_SIZE-1){
				if (labelTile[borderYLoc][borderXLoc] == targetLabel && 
				    intensityTile[borderYLoc][borderXLoc] >= lowerIntensityBound && 
				    intensityTile[borderYLoc][borderXLoc] <= upperIntensityBound){
					phiTile[borderYLoc][borderXLoc] = 1;
				} else if (intensityTile[borderYLoc][borderXLoc] >= lowerIntensityBound && 
					   intensityTile[borderYLoc][borderXLoc] <= upperIntensityBound) {
					phiTile[borderYLoc][borderXLoc] = 2;
				} else {
					phiTile[borderYLoc][borderXLoc] = 0;
				}
				if (labelTile[borderYLoc][borderXLoc+1] == targetLabel && 
				    intensityTile[borderYLoc][borderXLoc+1] >= lowerIntensityBound && 
				    intensityTile[borderYLoc][borderXLoc+1] <= upperIntensityBound){
					phiTile[borderYLoc][borderXLoc+1] = 1;
				} else if (intensityTile[borderYLoc][borderXLoc+1] >= lowerIntensityBound && 
					   intensityTile[borderYLoc][borderXLoc+1] <= upperIntensityBound) {
					phiTile[borderYLoc][borderXLoc+1] = 2;
				} else {
					phiTile[borderYLoc][borderXLoc+1] = 0;
				}
			}
		}
		
	}
	
	// Update vertical border
	x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	
	borderYLoc = (ty<<TTSB)+1;
	if (tx == 0 ){		
		// Location to write in shared memory
		borderXLoc = 0;
		if (bx != 0) {
			// Upper block border
			x-=THREAD_TILE_SIZE;
			shiftTileReg1 = 8;
			shiftTileReg2 = 24;
		}
	} else if (tx == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderXLoc = TILE_SIZE+1;			
		if (bx != gridDim.x-1) {
			// Lower block border
			x+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 16;
		}
	}
	// Read from global and write to shared memory
	if (tx == 0 || tx == BLOCK_TILE_SIZE-1) {
		if ((bx == 0           && tx == 0                ) || 
		    (bx == gridDim.x-1 && tx == BLOCK_TILE_SIZE-1)){
			intensityTile[borderYLoc  ][borderXLoc] = 0;
			intensityTile[borderYLoc+1][borderXLoc] = 0;
			    labelTile[borderYLoc  ][borderXLoc] = 0;
			    labelTile[borderYLoc+1][borderXLoc] = 0;
			      phiTile[borderYLoc  ][borderXLoc] = 0;
			      phiTile[borderYLoc+1][borderXLoc] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                ) |
				    (((y>>TTSB)&BTSMask) << BTSB        ) |
				     ((x>>TSB)            << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderIntData = intensity[bLocation];
			borderLabData =    labels[bLocation];
			
			intensityTile[borderYLoc][borderXLoc  ] 
					= ( borderIntData >> shiftTileReg1 ) & 0xFF;
			intensityTile[borderYLoc+1][borderXLoc] 
					= ( borderIntData >> shiftTileReg2 ) & 0xFF;
			    labelTile[borderYLoc][borderXLoc  ] 
					= ( borderLabData >> shiftTileReg1 ) & 0xFF;
			    labelTile[borderYLoc+1][borderXLoc] 
					= ( borderLabData >> shiftTileReg2 ) & 0xFF;
			// Update Phi of border
			if (tx == 0 || tx == BLOCK_TILE_SIZE-1){
				if (labelTile[borderYLoc][borderXLoc] == targetLabel && 
				    intensityTile[borderYLoc][borderXLoc] >= lowerIntensityBound && 
				    intensityTile[borderYLoc][borderXLoc] <= upperIntensityBound){
					phiTile[borderYLoc][borderXLoc] = 1;
				} else if (intensityTile[borderYLoc][borderXLoc] >= lowerIntensityBound && 
					   intensityTile[borderYLoc][borderXLoc] <= upperIntensityBound) {
					phiTile[borderYLoc][borderXLoc] = 2;
				} else {
					phiTile[borderYLoc][borderXLoc] = 0;
				}
				if (labelTile[borderYLoc+1][borderXLoc] == targetLabel && 
				    intensityTile[borderYLoc+1][borderXLoc] >= lowerIntensityBound && 
				    intensityTile[borderYLoc+1][borderXLoc] <= upperIntensityBound){
					phiTile[borderYLoc+1][borderXLoc] = 1;
				} else if (intensityTile[borderYLoc+1][borderXLoc] >= lowerIntensityBound && 
					   intensityTile[borderYLoc+1][borderXLoc] <= upperIntensityBound) {
					phiTile[borderYLoc+1][borderXLoc] = 2;
				} else {
					phiTile[borderYLoc+1][borderXLoc] = 0;
				}
			}
		}
	}	
	localGBI = 0;
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////////////////////
	
	// Initialization
	for (int tempY = ty+1; tempY <= TILE_SIZE; tempY+=BLOCK_TILE_SIZE ){
		for (int tempX = tx+1; tempX <= TILE_SIZE; tempX+=BLOCK_TILE_SIZE ){
			
			int ownIntData = intensityTile[tempY][tempX];
			if(ownIntData >= lowerIntensityBound && 
			   ownIntData <= upperIntensityBound) {
				if (labelTile[tempY][tempX] == targetLabel)
					phiTile[tempY][tempX] = 1;
				else {
					localGBI=1;//localGBI++;
					phiTile[tempY][tempX] = 2;
				}
			} else {
				phiTile[tempY][tempX] = 0;
			}
		}
	}
	__syncthreads();

	// Write back to main memory
	int phiData1 = phiTile[sharedY  ][sharedX  ] & 0xFF;
        int phiData2 = phiTile[sharedY  ][sharedX+1] & 0xFF;
        int phiData3 = phiTile[sharedY+1][sharedX  ] & 0xFF;
        int phiData4 = phiTile[sharedY+1][sharedX+1] & 0xFF;

        int phiReturnData = phiData1        |
			   (phiData2 << 8 ) |
			   (phiData3 << 16) |
			   (phiData4 << 24);
			
	phi[location] = phiReturnData;
	
	if (tx == 0 && ty == 0){
		globalBlockIndicator[blockId]=localGBI;
	}
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
	__shared__ signed char    phiTile[TILE_SIZE+2][TILE_SIZE+2]; // input/output

	// Flags
	__shared__ volatile char BlockChange;
	__shared__ volatile char change;
	__shared__ volatile int redoBlock;
		
	// Read Global Block Indicator from global memory
	int localGBI = globalBlockIndicator[blockId];
	
	// Set Block Variables
	redoBlock = 0;
	
	__syncthreads();
	
	if (localGBI > 0) {
		
		// Read Input Data into Shared Memory
		/////////////////////////////////////////////////////////////////////////////////////

		int x = bx<<BTSB;
		x = x + tx;
		x = x<<TTSB;
		int y = by<<BTSB;
		y = y + ty;
		y = y<<TTSB;
		  
		int location = 	(((x>>TTSB)&BTSMask)                ) |
				(((y>>TTSB)&BTSMask) << BTSB        ) |
				((x>>TSB)            << (BTSB+BTSB) ) ;
		location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
		int phiData = phi[location];
		
		int sharedX = tx*THREAD_TILE_SIZE+1;
		int sharedY = ty*THREAD_TILE_SIZE+1;
		
		phiTile[sharedY  ][sharedX  ] = phiData         & 0xFF;
		phiTile[sharedY  ][sharedX+1] = (phiData >>  8) & 0xFF;
		phiTile[sharedY+1][sharedX  ] = (phiData >> 16) & 0xFF;
		phiTile[sharedY+1][sharedX+1] = (phiData >> 24) & 0xFF;

		// Read Border Data into Shared Memory
		/////////////////////////////////////////////////////////////////////////////////////
		
		// Registers meant for speed. Two given each thread will update 2 pixels.
		int shiftTileReg1 = 0;
		int shiftTileReg2 = 0;
		
		int borderXLoc = 0;
		int borderYLoc = 0;
		
		// Needed Variables
		int bLocation;
		int borderPhiData;
		
		// Update horizontal border
		borderXLoc = sharedX;
		if (ty == 0 ){		
			// Location to write in shared memory
			borderYLoc = 0;
			if (by != 0) {
				// Upper block border
				y-=THREAD_TILE_SIZE;
				shiftTileReg1 = 16;
				shiftTileReg2 = 24;
			}
		} else if (ty == BLOCK_TILE_SIZE-1){
			// Location to write in shared memory
			borderYLoc = TILE_SIZE+1;			
			if (by != gridDim.y-1) {
				// Lower block border
				y+=THREAD_TILE_SIZE;
				shiftTileReg1 = 0;
				shiftTileReg2 = 8;
			}
		}
		// Read from global and write to shared memory
		if (ty == 0 || ty == BLOCK_TILE_SIZE-1) {
			if ((by == 0           && ty == 0                ) || 
			    (by == gridDim.y-1 && ty == BLOCK_TILE_SIZE-1)){
				phiTile[borderYLoc][borderXLoc  ] = 0;
				phiTile[borderYLoc][borderXLoc+1] = 0;
			} else {
				bLocation = (((x>>TTSB)&BTSMask)                 ) |
					    (((y>>TTSB)&BTSMask)  << BTSB        ) |
					     ((x>>TSB)            << (BTSB+BTSB) ) ;
				bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
				
				borderPhiData = phi[bLocation];
				
				phiTile[borderYLoc][borderXLoc  ] 
						= ( borderPhiData >> shiftTileReg1 ) & 0xFF;
				phiTile[borderYLoc][borderXLoc+1] 
						= ( borderPhiData >> shiftTileReg2 ) & 0xFF;
			}
		}
			
		// Update vertical border
		x = bx<<BTSB;
		x = x + tx;
		x = x<<TTSB;
		y = by<<BTSB;
		y = y + ty;
		y = y<<TTSB;
		
		borderYLoc = sharedY;
		if (tx == 0 ){		
			// Location to write in shared memory
			borderXLoc = 0;
			if (bx != 0) {
				// Upper block border
				x-=THREAD_TILE_SIZE;
				shiftTileReg1 = 8;
				shiftTileReg2 = 24;
			}
		} else if (tx == BLOCK_TILE_SIZE-1){
			// Location to write in shared memory
			borderXLoc = TILE_SIZE+1;			
			if (bx != gridDim.x-1) {
				// Lower block border
				x+=THREAD_TILE_SIZE;
				shiftTileReg1 = 0;
				shiftTileReg2 = 16;
			}
		}
		// Read from global and write to shared memory
		if (tx == 0 || tx == BLOCK_TILE_SIZE-1) {
			if ((bx == 0           && tx == 0                ) || 
			    (bx == gridDim.x-1 && tx == BLOCK_TILE_SIZE-1)){
				phiTile[borderYLoc][borderXLoc  ] = 0;
				phiTile[borderYLoc+1][borderXLoc] = 0;
			} else {
				bLocation = (((x>>TTSB)&BTSMask)                 ) |
					    (((y>>TTSB)&BTSMask)  << BTSB        ) |
					     ((x>>TSB)            << (BTSB+BTSB) ) ;
				bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
				
				borderPhiData = phi[bLocation];
				
				phiTile[borderYLoc][borderXLoc  ] 
						= ( borderPhiData >> shiftTileReg1 ) & 0xFF;
				phiTile[borderYLoc+1][borderXLoc] 
						= ( borderPhiData >> shiftTileReg2 ) & 0xFF;
			}
		}
		
		BlockChange = 0; // Shared variable
		change       = 1; // Shared variable
		__syncthreads();
		
		// Algorithm 
		/////////////////////////////////////////////////////////////////////

		while (change){
			__syncthreads();
			change = 0;
			__syncthreads();
			
			for (int tempY = ty+1; tempY <= TILE_SIZE; tempY+=BLOCK_TILE_SIZE ){
				for (int tempX = tx+1; tempX <= TILE_SIZE; tempX+=BLOCK_TILE_SIZE ){
					
					if((phiTile[tempY+1][tempX  ]  == 1 ||
					    phiTile[tempY-1][tempX  ]  == 1 ||
					    phiTile[tempY  ][tempX-1]  == 1 ||
					    phiTile[tempY  ][tempX+1]  == 1 ) && 
					    phiTile[tempY  ][tempX  ]  == 2){
						phiTile  [tempY][tempX] = 1;
						change = 1;
						BlockChange = 1;
					}
				}
			}
			__syncthreads();
		}
		
		// To check if there are pixels with phi value of 2
		for (int tempY = ty+1; tempY <= TILE_SIZE; tempY+=BLOCK_TILE_SIZE ){
			for (int tempX = tx+1; tempX <= TILE_SIZE; tempX+=BLOCK_TILE_SIZE ){
				if (phiTile[tempY][tempX] == 2){
					redoBlock = 1;
				}
			}
		}
		
		__syncthreads();
		
		int phiData1 = phiTile[sharedY  ][sharedX  ] & 0xFF;
		int phiData2 = phiTile[sharedY  ][sharedX+1] & 0xFF;
		int phiData3 = phiTile[sharedY+1][sharedX  ] & 0xFF;
		int phiData4 = phiTile[sharedY+1][sharedX+1] & 0xFF;

		int phiReturnData = phiData1        |
				   (phiData2 << 8 ) |
				   (phiData3 << 16) |
				   (phiData4 << 24);
				
		phi[location] = phiReturnData;
			
		if (tx == 0 && ty == 0) {
			if (BlockChange){
				*globalFinishedVariable = 1;
				__threadfence();
			}
			globalBlockIndicator[blockId] = redoBlock;
		}
	}
}

/*
 * Lss Step 3 from Pseudo Code
 */
__global__ void lssStep3(signed int* phi,
			 signed int* phiOut) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
				
	// Including border
	__shared__ signed char    phiTile[TILE_SIZE+2][TILE_SIZE+2]; // input
	__shared__ signed char phiOutTile[TILE_SIZE+2][TILE_SIZE+2]; // output

	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	int x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	int y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	  
	int location = 	(((x>>TTSB)&BTSMask)                ) |
			(((y>>TTSB)&BTSMask) << BTSB        ) |
			((x>>TSB)            << (BTSB+BTSB) ) ;
	location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
		
	int phiData = phi[location];
	
	int sharedX = tx*THREAD_TILE_SIZE+1;
	int sharedY = ty*THREAD_TILE_SIZE+1;
	
	phiTile[sharedY  ][sharedX  ] = phiData         & 0xFF;
	phiTile[sharedY  ][sharedX+1] = (phiData >>  8) & 0xFF;
	phiTile[sharedY+1][sharedX  ] = (phiData >> 16) & 0xFF;
	phiTile[sharedY+1][sharedX+1] = (phiData >> 24) & 0xFF;

	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////
	
	// Registers meant for speed. Two given each thread will update 2 pixels.
	int shiftTileReg1 = 0;
	int shiftTileReg2 = 0;
	
	int borderXLoc = 0;
	int borderYLoc = 0;
	
	// Needed Variables
	int bLocation;
	int borderPhiData;
	
	// Update horizontal border
	borderXLoc = sharedX;
	if (ty == 0 ){		
		// Location to write in shared memory
		borderYLoc = 0;
		if (by != 0) {
			// Upper block border
			y-=THREAD_TILE_SIZE;
			shiftTileReg1 = 16;
			shiftTileReg2 = 24;
		}
	} else if (ty == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderYLoc = TILE_SIZE+1;			
		if (by != gridDim.y-1) {
			// Lower block border
			y+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 8;
		}
	}
	// Read from global and write to shared memory
	if (ty == 0 || ty == BLOCK_TILE_SIZE-1) {
		if ((by == 0           && ty == 0                ) || 
		    (by == gridDim.y-1 && ty == BLOCK_TILE_SIZE-1)){
			phiTile[borderYLoc][borderXLoc  ] = 0;
			phiTile[borderYLoc][borderXLoc+1] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                 ) |
				    (((y>>TTSB)&BTSMask)  << BTSB        ) |
				     ((x>>TSB)            << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderPhiData = phi[bLocation];
			
			phiTile[borderYLoc][borderXLoc  ] 
					= ( borderPhiData >> shiftTileReg1 ) & 0xFF;
			phiTile[borderYLoc][borderXLoc+1] 
					= ( borderPhiData >> shiftTileReg2 ) & 0xFF;
		}
	}
		
	// Update vertical border
	x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	
	borderYLoc = sharedY;
	if (tx == 0 ){		
		// Location to write in shared memory
		borderXLoc = 0;
		if (bx != 0) {
			// Upper block border
			x-=THREAD_TILE_SIZE;
			shiftTileReg1 = 8;
			shiftTileReg2 = 24;
		}
	} else if (tx == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderXLoc = TILE_SIZE+1;			
		if (bx != gridDim.x-1) {
			// Lower block border
			x+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 16;
		}
	}
	// Read from global and write to shared memory
	if (tx == 0 || tx == BLOCK_TILE_SIZE-1) {
		if ((bx == 0           && tx == 0                ) || 
		    (bx == gridDim.x-1 && tx == BLOCK_TILE_SIZE-1)){
			phiTile[borderYLoc][borderXLoc  ] = 0;
			phiTile[borderYLoc+1][borderXLoc] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                 ) |
				    (((y>>TTSB)&BTSMask)  << BTSB        ) |
				     ((x>>TSB)            << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderPhiData = phi[bLocation];
			
			phiTile[borderYLoc][borderXLoc  ] 
					= ( borderPhiData >> shiftTileReg1 ) & 0xFF;
			phiTile[borderYLoc+1][borderXLoc] 
					= ( borderPhiData >> shiftTileReg2 ) & 0xFF;
		}
	}
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////////////////////
	
	for (int tempY = ty+1; tempY <= TILE_SIZE; tempY+=BLOCK_TILE_SIZE ){
		for (int tempX = tx+1; tempX <= TILE_SIZE; tempX+=BLOCK_TILE_SIZE ){
			
			if(phiTile[tempY][tempX] == 1) {
				if(phiTile[tempY+1][tempX]  == 1 &&
				   phiTile[tempY-1][tempX]  == 1 &&
				   phiTile[tempY][tempX+1]  == 1 &&
				   phiTile[tempY][tempX-1]  == 1 ){
					phiOutTile[tempY][tempX] = 0xFD;
				} else 
					phiOutTile[tempY][tempX] = 0xFF;
			} else
				if(phiTile[tempY+1][tempX]  == 1 ||
				   phiTile[tempY-1][tempX]  == 1 ||
				   phiTile[tempY][tempX+1]  == 1 ||
				   phiTile[tempY][tempX-1]  == 1 ){
					phiOutTile[tempY][tempX] = 1;
				} else 
					phiOutTile[tempY][tempX] = 3;
		}
	}
	
	__syncthreads();

	// Write back to main memory
	int phiData1 = phiOutTile[sharedY  ][sharedX  ] & 0xFF;
        int phiData2 = phiOutTile[sharedY  ][sharedX+1] & 0xFF;
        int phiData3 = phiOutTile[sharedY+1][sharedX  ] & 0xFF;
        int phiData4 = phiOutTile[sharedY+1][sharedX+1] & 0xFF;

        int phiReturnData = phiData1        |
			   (phiData2 << 8 ) |
			   (phiData3 << 16) |
			   (phiData4 << 24);
	
	phiOut[location] = phiReturnData;

}

__global__ void evolveContour(unsigned int* intensity, 
			      unsigned int* labels,
			      signed int* phi,
			      signed int* phiOut, 
			      int gridXSize,
			      int gridYSize,
			      int* targetLabels, 
			      int* lowerIntensityBounds, 
			      int* upperIntensityBounds,
			      int max_iterations, 
			      int* globalBlockIndicator,
			      int* globalFinishedVariable,
			      int* totalIterations ) {
        int tid = threadIdx.x;
	
	// Total iterations
	totalIterations = &totalIterations[tid];
	
	// Size in ints
	int size = (gridXSize*gridYSize)<<(BTSB+BTSB);
	
	// New phi pointer for each label.
	phi    = &phi[tid*size];
	phiOut = &phiOut[tid*size];

	globalBlockIndicator = &globalBlockIndicator[tid*gridXSize*gridYSize];

	// Global synchronization variable
	globalFinishedVariable = &globalFinishedVariable[tid];
	*globalFinishedVariable = 0;

	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Initialize phi array
	lssStep1<<<dimGrid, dimBlock>>>(intensity, 
					labels,  
					phi, 
					targetLabels[tid], 
					lowerIntensityBounds[tid], 
					upperIntensityBounds[tid],
					globalBlockIndicator,
					globalFinishedVariable );
	
	int iterations = 0;
	do {
		iterations++;
		lssStep2<<<dimGrid, dimBlock>>>(phi, 
					globalBlockIndicator,
					globalFinishedVariable );
		cudaDeviceSynchronize();
	} while (atomicExch(globalFinishedVariable,0) && (iterations < max_iterations));
	
	lssStep3<<<dimGrid, dimBlock>>>(phi,
					phiOut);
	
	*totalIterations = iterations;
}

signed int *levelSetSegment(unsigned int *intensity, 
			    unsigned int *labels,
			    int height, 
			    int width,
			    int *targetLabels, 
			    int *lowerIntensityBounds,
			    int *upperIntensityBounds,
			    int numLabels){
	
	#if defined(DEBUG)
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
	
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	#if defined(DEBUG)
		printf("\n Grid Size: %d %d\n", gridYSize, gridXSize);
		printf(  "Block Size: %d %d\n", BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	#endif
	
	int XSize = gridXSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	int YSize = gridYSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize*sizeof(char);
	
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
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.labels                 , gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.phi                    , numLabels*gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.phiOut                 , numLabels*gpu.size));
	checkCuda(cudaMalloc((void**)&gpu.globalBlockIndicator   , numLabels*gridXSize*gridYSize*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.globalFinishedVariable , numLabels*sizeof(int)));
	checkCuda(cudaMalloc((void**)&gpu.totalIterations        , numLabels*sizeof(int)));
	
	// Allocate result array in CPU memory
	gpu.phiOnCpu = new signed int[gpu.size*numLabels/sizeof(int)];
	gpu.totalIterationsOnCpu = new int [numLabels];
	
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
			gpu.size, 
			cudaMemcpyHostToDevice));
			
        checkCuda(cudaMemcpy(gpu.labels, 
			labels, 
			gpu.size, 
			cudaMemcpyHostToDevice));

	#if defined(KERNEL_TIMING)
		checkCuda(cudaDeviceSynchronize());
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
					gpu.phiOut, 
					gridXSize,
					gridYSize, 
					gpu.targetLabels, 
					gpu.lowerIntensityBounds, 
					gpu.upperIntensityBounds,
					max_iterations,
					gpu.globalBlockIndicator,
					gpu.globalFinishedVariable,
					gpu.totalIterations);
	

	#if defined(KERNEL_TIMING)
		checkCuda(cudaDeviceSynchronize());
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.phiOnCpu, 
			gpu.phiOut, 
			numLabels*gpu.size, 
			cudaMemcpyDeviceToHost));
	
	checkCuda(cudaMemcpy(gpu.totalIterationsOnCpu, 
			gpu.totalIterations, 
			numLabels*sizeof(int), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.labels));
	checkCuda(cudaFree(gpu.phi));
	checkCuda(cudaFree(gpu.phiOut));
	checkCuda(cudaFree(gpu.targetLabels));
	checkCuda(cudaFree(gpu.lowerIntensityBounds));
	checkCuda(cudaFree(gpu.upperIntensityBounds));
	checkCuda(cudaFree(gpu.globalBlockIndicator));
	checkCuda(cudaFree(gpu.globalFinishedVariable));
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif
	
	#if defined(VERBOSE)
		for (int i = 0; i < numLabels; i++){
			printf("target label: %d converged in %d iterations.\n", 
					targetLabels[i],
					gpu.totalIterationsOnCpu[i]);	
		}
	#endif
	
	return(gpu.phiOnCpu);

}
