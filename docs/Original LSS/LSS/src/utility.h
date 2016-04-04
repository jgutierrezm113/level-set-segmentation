#ifndef UTILITY_C_H
#define UTILITY_C_H

#include "image.h"

#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#ifndef BUF_SIZE
#define BUF_SIZE 256
#endif

class pnm_error { };

struct pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

void pnm_read(std::ifstream &file, char* buf);
Image<unsigned char>* loadPGM(const char* name);
void savePPM(Image<pixel> *im, const char *name);
pixel randomcolor();
void loadImages(unsigned char **images, int *height, int *width, char *imageFile, int numberOfRepeats);
void loadLabels(int **labels, int **lower, int **upper, int *numberOfLabels, char *labelsFile);
void saveImage(unsigned char *img, int height, int width, char *ouputName = (char *)"out.ppm");
void saveImages(unsigned char *images, int height, int width, int numberOfRepeats, int *labels, int numberOfLabels);
void saveImages(unsigned char *images, int height, int width, int numberOfRepeats, int *labels, int numberOfLabels, std::string output);

template<typename T>
void print(T* array, int HEIGHT, int WIDTH)
{
	for(int i=0; i<48; i++)
	{
		for(int j=0; j<48; j++)
			std::cout << std::setw(2) << (int) array[i*WIDTH + j];
		std::cout << std::endl;
	}
}

#endif
