#include "utility.h"
#include <fstream>

Image<unsigned char>* loadPGM(const char* name)
{
	char buf[BUF_SIZE];

	// Read header
	std::ifstream file(name, std::ios::in | std::ios::binary);
	pnm_read(file, buf);
	if (strncmp(buf, "P5", 2))
		throw pnm_error();

	pnm_read(file, buf);
	int width = atoi(buf);
	pnm_read(file, buf);
	int height = atoi(buf);

	pnm_read(file, buf);
	if(atoi(buf) > UCHAR_MAX)
		throw pnm_error();

	// Read data
	Image<unsigned char> *im = new Image<unsigned char>(width, height);
	file.read((char *)imPtr(im, 0, 0), width*height*sizeof(unsigned char));

	return im;
}

void pnm_read(std::ifstream &file, char* buf)
{
	char doc[BUF_SIZE];
	char c;

	file >> c;
	while (c == '#')
	{
		file.getline(doc, BUF_SIZE);
		file >> c;
	}
	file.putback(c);

	file.width(BUF_SIZE);
	file >> buf;
	file.ignore();
}

void savePPM(Image<pixel>* im, const char *name)
{
	int width = im->width();
	int height = im->height();
	std::ofstream file(name, std::ios::out | std::ios::binary);

	file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
	file.write((char *)imPtr(im, 0, 0), width*height*sizeof(pixel));
}

pixel randomcolor()
{
	pixel c;
	c.r = (unsigned char) rand();
	c.g = (unsigned char) rand();
	c.b = (unsigned char) rand();

	return c;
}

void loadImages(unsigned char **images, int *height, int *width, char *imageFile, int numberOfRepeats = 1)
{
	Image<unsigned char>* input = loadPGM(imageFile);
	const int HEIGHT = input->height();
	const int WIDTH = input->width();
	const int size = HEIGHT * WIDTH * sizeof(char);

	*images = new unsigned char[numberOfRepeats * HEIGHT * WIDTH];

	for(int i = 0; i < numberOfRepeats; i++)
		memcpy(&(*images)[i * HEIGHT * WIDTH], input->data, size);

	*height = HEIGHT;
	*width = WIDTH;
}

void loadLabels(int **labels, int **lower, int **upper, int *numberOfLabels, char *labelsFile)
{
	int n = 0;
	int c;
	FILE* handle = fopen(labelsFile, "r");
	do
	{
		c = getc(handle);
		if(c == '\n')
			n++;
	}
	while(c != EOF);
	n /= 3;
	fclose(handle);

	*labels = new int[n];
	*lower = new int[n];
	*upper = new int[n];
	*numberOfLabels = n;

	char line[8];
	std::ifstream ifs;
	ifs.open(labelsFile);
	for(int i = 0; i < n; i++)
	{
		ifs.getline(line, 8);
		*labels[i] = atoi(line);

		ifs.getline(line, 8);
		*lower[i] = atoi(line);

		ifs.getline(line, 8);
		*upper[i] = atoi(line);
	}
	ifs.close();
}

void saveImage(unsigned char *img, int height, int width, char *ouputName)
{
	srand(time(NULL));
	pixel color[height*width];
	for(int i = 0; i < 255; i++)
		color[i] = randomcolor();

	Image<pixel> *output = new Image<pixel>(width, height, true);
	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			output->access[i][j] = color[img[i * width + j]];

	savePPM(output, ouputName);
}

void saveImages(unsigned char *images, int height, int width, int numberOfRepeats, int *labels, int numberOfLabels)
{
	srand(time(NULL));
	pixel color[height*width];
	for(int i = 0; i < 255; i++)
			color[i] = randomcolor();

	char fileName[32];
	for(int l = 0; l < numberOfRepeats; l++)
	{
		for(int k = 0; k < numberOfLabels; k++)
		{
			Image<pixel> *im = new Image<pixel>(width, height, true);
			for(int i=0; i < height; i++)
				for(int j=0; j < width; j++)
					im->access[i][j] = color[images[(l * numberOfLabels + k) * height * width + i * width + j]];

			strcpy(fileName, "out");
			sprintf(&fileName[3], "%d-%d", l, labels[k]);
			strcat(fileName, ".ppm");
			savePPM(im, fileName);
		}
	}
}

void saveImages(unsigned char *images, int height, int width, int numberOfRepeats, int *labels, int numberOfLabels, std::string output)
{
	srand(time(NULL));
	pixel color[height*width];
	for(int i = 0; i < 255; i++)
			color[i] = randomcolor();

	char fileName[32];
	for(int l = 0; l < numberOfRepeats; l++)
	{
		for(int k = 0; k < numberOfLabels; k++)
		{
			Image<pixel> *im = new Image<pixel>(width, height, true);
			for(int i=0; i < height; i++)
				for(int j=0; j < width; j++)
					im->access[i][j] = color[images[(l * numberOfLabels + k) * height * width + i * width + j]];

			strcpy(fileName, "out");
			sprintf(&fileName[3], "%d-%d", l, labels[k]);
			strcat(fileName, ".ppm");
			std::string outputFile;
			if (output.length() != 0 && output[output.length() - 1] != '/')
			{
				outputFile = output + "/" + fileName;
			}
			else
			{
				outputFile = output + fileName;
			}
			savePPM(im, outputFile.c_str());
		}
	}
}
