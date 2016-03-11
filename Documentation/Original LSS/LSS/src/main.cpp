#include "image.h"
#include "utility.h"

#include <iostream>

using namespace std;

extern signed char *levelSetSegment(unsigned char *intensity, unsigned char *component, int height, int width, int numberOfImages, int *labels, int numberOfLabels);

int main(int argc, char** argv)
{
	// Set inputs
	char *intensityImage = (char *)"../resource/Intensities2.pgm";
	char *componentImage = (char *)"../resource/Components2.pgm";
	char *output = (char *)"../output";
	int labels[5] = {1, 2, 3, 4, 5};
	int numberOfRepeats = 101;
	int numberOfLabels = 5;

	// Load intensity & component image
	unsigned char *intensity, *component;
	int height, width;
	loadImages(&intensity, &height, &width, intensityImage, numberOfRepeats);
	loadImages(&component, &height, &width, componentImage, numberOfRepeats);

	signed char *phi = levelSetSegment(intensity, component, height, width, numberOfRepeats, labels, numberOfLabels);

	saveImages((unsigned char *)phi, height, width, numberOfRepeats, labels, numberOfLabels, output);

	cout << "done!" << endl;
}
