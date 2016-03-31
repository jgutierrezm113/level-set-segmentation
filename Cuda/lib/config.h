/* 
 * Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 *   
 */
 
#ifndef CONFIG_H
#define CONFIG_H

// Defining verbosity of run
//#define VERBOSE

// Defining important constants
#define MAX_LABELS_PER_IMAGE 32

// Threshold of max number of iterations of analysis
#define MAX_ITER 5000

// timing directives
#define CUDA_TIMING

// Debug Cuda Errors
#define DEBUG

/* 
 * Each thread will handle 4 chars (4 bytes)
 * Values should be powers of 2.
 */
#define THREAD_TILE_SIZE 1
#define TTSMask	0 	// THREAD_TILE_SIZE-1
#define TTSB 1 

#define BLOCK_TILE_SIZE 32
#define BTSMask 15 	// BLOCK_TILE_SIZE-1
#define BTSB 4

#define TILE_SIZE 32 	// THREAD_TILE_SIZE*BLOCK_TILE_SIZE
#define TSMask 31 	// TILE_SIZE-1
#define TSB 5 		// TTSB+BTSB

#endif

