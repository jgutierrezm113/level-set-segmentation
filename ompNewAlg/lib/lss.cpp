/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Level Set Segmentation for Image Processing 
 *  
 */
 
#include "lss.h"

using namespace std;

void modMaxIter (int value){
	max_iterations = value;
}

void evolveContour(unsigned char* intensity, 
		   unsigned char* labels, 
		   signed char* phi, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels,
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j) {

	// Note: j is the label counter
        phi       = &phi      [j*HEIGHT*WIDTH];
	
	char *phiTemp = new char [HEIGHT*WIDTH];
		
	// Timing variables
	double processing_tbegin;
	double processing_tend;
	
	double total_tbegin = omp_get_wtime();
	double total_tend;
	
	double initializing_tbegin = omp_get_wtime();
	double initializing_tend;

	
	// Step 1: Initialize
	#pragma omp parallel for
	for (int k = 0; k < HEIGHT*WIDTH; k++){
		
		// Note: Using x and y so its easier to map into 2D array (row mayor)
		int xPos = k%WIDTH;
		int yPos = k/WIDTH;
				
		if (intensity[yPos*WIDTH+xPos] >= lowerIntensityBounds[j] &&
		    intensity[yPos*WIDTH+xPos] <= upperIntensityBounds[j]){
			  if (labels[yPos*WIDTH+xPos] == targetLabels[j]){
				phiTemp[yPos*WIDTH+xPos] = 1;
			  }  else {
				phiTemp[yPos*WIDTH+xPos] = 2;				  
			  }
		} else {
			phiTemp[yPos*WIDTH+xPos] = 0;
		}		
	}
	
	initializing_tend = omp_get_wtime();
	double elapsed_secs = initializing_tend - initializing_tbegin;
	
	/* Continue */
        int iterations = 0;
	int globalFinishedVariable;
	processing_tbegin = omp_get_wtime();
	
	
	// Step 2: Evolution
	do {
		globalFinishedVariable = 0;
		iterations++;
		#pragma omp parallel for
		for (int k = 0; k < HEIGHT*WIDTH; k++){
			
			// Note: Using x and y so its easier to map into 2D array (row mayor)
			int xPos = k%WIDTH;
			int yPos = k/WIDTH;
			
			// Get border values
			char borderUp;
			char borderDown;
			char borderLeft;
			char borderRight;
			
			if (xPos > 0){
				borderLeft = phiTemp[yPos*WIDTH+(xPos-1)];
			} else {
				borderLeft = 0;
			}
			
			if (xPos < WIDTH-1){
				borderRight = phiTemp[yPos*WIDTH+(xPos+1)];
			} else {
				borderRight = 0;
			}
			
			if (yPos > 0){
				borderDown = phiTemp[(yPos-1)*WIDTH+xPos];
			} else {
				borderDown = 0;
			}
			
			if (yPos < HEIGHT-1){
				borderUp = phiTemp[(yPos+1)*WIDTH+xPos];
			} else {
				borderUp = 0;
			}

			// Algorithm
			if((borderUp     == 1 ||
			    borderDown   == 1 ||
			    borderLeft   == 1 ||
			    borderRight  == 1 ) && 
			    phiTemp[yPos*WIDTH+xPos]  == 2){
				phiTemp[yPos*WIDTH+xPos] = 1;
				globalFinishedVariable = 1;
			}
		}
	} while (globalFinishedVariable && (iterations < max_iterations));
	
		
	// Step 3: Finalize
	#pragma omp parallel for
	for (int k = 0; k < HEIGHT*WIDTH; k++){
		
		
		// Note: Using x and y so its easier to map into 2D array (row mayor)
		int xPos = k%WIDTH;
		int yPos = k/WIDTH;
		
		// Get border values
		char borderUp;
		char borderDown;
		char borderLeft;
		char borderRight;
		
		if (xPos > 0){
			borderLeft = phiTemp[yPos*WIDTH+(xPos-1)];
		} else {
			borderLeft = 0;
		}
		
		if (xPos < WIDTH-1){
			borderRight = phiTemp[yPos*WIDTH+(xPos+1)];
		} else {
			borderRight = 0;
		}
		
		if (yPos > 0){
			borderDown = phiTemp[(yPos-1)*WIDTH+xPos];
		} else {
			borderDown = 0;
		}
		
		if (yPos < HEIGHT-1){
			borderUp = phiTemp[(yPos+1)*WIDTH+xPos];
		} else {
			borderUp = 0;
		}
				
		if (phiTemp[yPos*WIDTH+xPos] == 1){
			if(borderUp     == 1 &&
			   borderDown   == 1 &&
			   borderLeft   == 1 &&
			   borderRight  == 1 ){
				phi[yPos*WIDTH+xPos] = -3;
			} else 
				phi[yPos*WIDTH+xPos] = -1;
		} else {
			if(borderUp     == 1 ||
			   borderDown   == 1 ||
			   borderLeft   == 1 ||
			   borderRight  == 1 ){
				phi[yPos*WIDTH+xPos] = 1;
			} else 
				phi[yPos*WIDTH+xPos] = 3;		
		}
	}

	printf("Initializing Time: %f\n", elapsed_secs);
	
	processing_tend = omp_get_wtime();
	elapsed_secs = processing_tend - processing_tbegin;
	
	printf("Processing Time: %f\n", elapsed_secs);
	
	total_tend = omp_get_wtime();
	elapsed_secs = total_tend - total_tbegin;
	
	printf("Total Time: %f\n", elapsed_secs);
	
	free (phiTemp);
}