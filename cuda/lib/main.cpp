/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Level Set Segmentation for Image Processing 
 *  
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#include "imghandler.h"
#include "config.h"

typedef struct __cpuData {
	
	int height;
	int width;
	int gridXSize;
	int gridYSize;
	int size;	

	unsigned char* intensity;
	unsigned char* labels;
	
	signed int* result;
	
	int targetLabels[MAX_LABELS_PER_IMAGE];
	int lowerIntensityBounds[MAX_LABELS_PER_IMAGE];
	int upperIntensityBounds[MAX_LABELS_PER_IMAGE];
	int numLabels;
	
} cpuData;

cpuData cpu;

extern signed int *levelSetSegment(unsigned int *intensity, 
				   unsigned int *labels, 
				   int height, 
				   int width, 
				   int *targetLabels, 
				   int *lowerIntensityBounds,
				   int *upperIntensityBounds,
				   int numLabels);
			    
extern void modMaxIter (int value);

int main(int argc, char* argv[]) {

	// Files needed
	char* imageFile = NULL;
	char* labelFile = NULL;
	char* paramFile = NULL;

	for(int i = 1 ; i < argc ; i++) {
		if(strcmp(argv[i], "--image") == 0) {
			if(i + 1 < argc)
				imageFile = argv[++i];
		} else if(strcmp(argv[i], "--labels") == 0) {
			if(i + 1 < argc)
				labelFile = argv[++i];
		} else if(strcmp(argv[i], "--params") == 0) {
			if(i + 1 < argc)
				paramFile = argv[++i];
		} else if(strcmp(argv[i], "--max_reps") == 0) {
			if(i + 1 < argc)
				modMaxIter(atoi(argv[++i]));
		}
	}
	
	if(imageFile == NULL || labelFile == NULL || paramFile == NULL) {
		cerr << "Missing one or more arguments. " << endl;
		exit(1);
	}

	bool produceOutput = true;

        // Load Intensity Image
	image<unsigned char>* intensityInput = loadPGM(imageFile);
	
        // Load connected component labels
	image<unsigned char>* labelInput     = loadPGM(labelFile);
	
	cpu.height = intensityInput->height();
	cpu.width  = intensityInput->width();
	
	cpu.gridXSize = 1 + (( cpu.width - 1) / TILE_SIZE);
	cpu.gridYSize = 1 + ((cpu.height - 1) / TILE_SIZE);
	
	int XSize = cpu.gridXSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	int YSize = cpu.gridYSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	
	cpu.size = XSize*YSize;

	cpu.intensity = new unsigned char[cpu.size];
	cpu.labels    = new unsigned char[cpu.size];
		
	for (int y = 0 ; y < YSize ; y++){
		for (int x = 0 ; x < XSize ; x++){
			int newLocation =  ( x&TTSMask )				  |
					  (( y&TTSMask ) 	  <<  TTSB		) |
					 ((( x>>TTSB ) &BTSMask ) << (TTSB+TTSB)	) |
					 ((( y>>TTSB ) &BTSMask ) << (BTSB+TTSB+TTSB)	) |
					  (( x>>TSB  )            << (TSB+TSB) 	) ;
			newLocation += 	  (( y>>TSB  )		  << (TSB+TSB) 	)*cpu.gridXSize;
			
			if (x < cpu.width && y < cpu.height) {
				cpu.intensity[newLocation] = intensityInput->data[y*cpu.width + x];
				cpu.labels[newLocation]    =     labelInput->data[y*cpu.width + x];
			} else{
				// Necessary in case image size is not a multiple of TILE_SIZE
				cpu.intensity[newLocation] = 0;
				cpu.labels[newLocation]    = 0;
			}
		}
	}
	
	// Load parameters from parameter file
	ifstream paramStream;
	paramStream.open(paramFile);

	if(paramStream.is_open() != true) {
		cerr << "Could not open '" << paramFile << "'." << endl;
		exit(1);
	}
	
	cpu.numLabels = 0;
	
	while(paramStream.eof() == false) {
		char line[16];
		paramStream.getline(line, 16);
		
		if(paramStream.eof() == true)
			break;

		if(cpu.numLabels % 3 == 0)
			cpu.targetLabels[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		else if(cpu.numLabels % 3 == 1)
			cpu.lowerIntensityBounds[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		else
			cpu.upperIntensityBounds[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		
		cpu.numLabels++;
	}
	
	if(cpu.numLabels % 3 == 0) {
		cpu.numLabels /= 3;
		if(cpu.numLabels > MAX_LABELS_PER_IMAGE){
			cerr << "Exceeded maximum number of objects per image." << endl;
			exit(1);
		}
	} else {
		cerr << "Number of lines in " << paramFile << " is not divisible by 3 and it should. " << endl;
		exit(1);
	}
	paramStream.close();

	#if defined(VERBOSE)
		cout << "Finished Processing files with " << cpu.numLabels 
		     << " Objects in the image of size " << cpu.height
		     << "x" << cpu.width << endl;
	#endif

	cpu.result = levelSetSegment((unsigned int *)cpu.intensity, 
				     (unsigned int *)cpu.labels,
				     cpu.height, 
				     cpu.width, 
				     cpu.targetLabels, 
				     cpu.lowerIntensityBounds,
				     cpu.upperIntensityBounds,
				     cpu.numLabels);
	
	// Output RGB images
	if(produceOutput == true) {
		
		char filename[64];
		sprintf(filename, "result.ppm");
		
		#if defined(VERBOSE)
			cout << "Produccing final image file " << filename << "." << endl;
		#endif
		
		Color color;
		
		// Create 1 image for all labels
		image<Color> output = image<Color>(cpu.width, cpu.height, true);
		image<Color>* im = &output;
			
		for(int k = 0 ; k < cpu.numLabels ; k++) {
			Color randomcolor = randomColor();
			for (int y = 0 ; y < cpu.height ; y++){
				for (int x = 0 ; x < cpu.width ; x++){
					int newLocation =  ( x&TTSMask )				  |
							  (( y&TTSMask )          <<  TTSB		) |
							 ((( x>>TTSB ) &BTSMask ) << (TTSB+TTSB)	) |
							 ((( y>>TTSB ) &BTSMask ) << (BTSB+TTSB+TTSB)	) |
							  (( x>>TSB  )            << (TSB+TSB) 	) ;
					newLocation += 	  (( y>>TSB  )		  << (TSB+TSB) 	)*cpu.gridXSize;
					
					if (k == 0) {
						// Initialize output image as input.
						color.r = (char)cpu.intensity[newLocation];
						color.g = (char)cpu.intensity[newLocation];
						color.b = (char)cpu.intensity[newLocation];
						im->access[y][x] = color;
					}
					char *address_as_char = (char *)cpu.result;
					if (address_as_char[k*cpu.size+newLocation] == -1){
						color = randomcolor;
						im->access[y][x] = color;
					} else if (address_as_char[k*cpu.size+newLocation] == -3){
						color.r = randomcolor.r + OUTPUT_BRIGHTNESS;
						color.g = randomcolor.g + OUTPUT_BRIGHTNESS;
						color.b = randomcolor.b + OUTPUT_BRIGHTNESS;
						im->access[y][x] = color;
					}					
				}
			}
		}
		savePPM(im, filename);
	}
	
	// Free resources and end the program
	free(cpu.intensity);
	free(cpu.labels);

        return 0;
}