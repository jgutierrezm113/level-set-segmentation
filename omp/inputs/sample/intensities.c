#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
  size_t         width;
  size_t         height;
  unsigned char *data;
} Image;

static Image * image_new (size_t width, size_t height) {
	Image *image;

	image = malloc (sizeof *image);
	image->width = width;
	image->height = height;
	image->data = malloc (width * height);

	return image;
}

static void image_free (Image *image) {
	free (image->data);
	free (image);
}

static void image_fill (Image *image, unsigned char  value) {
	memset (image->data, value, image->width * image->height);
}

/**
 * image_set_pixel:
 *
 * Sets a pixel passed in signed (x, y) coordinates, where (0,0) is at
 * (x0,y0).
 **/
static void image_set_pixel (Image *image, int x0, int y0, ssize_t x, ssize_t y, unsigned char  value) {

	ssize_t tx = x0 + x;
	ssize_t ty = y0 + y;
	
	unsigned char *p;

	p = image->data + (ty * image->width) + tx;
	*p = value;
}

static void image_save (const Image *image, const char  *filename) {
	FILE *out;

	out = fopen (filename, "wb");
	if (!out)
	return;

	fprintf (out, "P5\n");
	fprintf (out, "%zu %zu\n", image->width, image->height);
	fprintf (out, "255\n");

	fwrite (image->data, 1, image->width * image->height, out);

	fclose (out);
}

static void draw_circle (Image *image, int x0, int y0, int radius, unsigned char  value) {
	int x, y;

	for (y = -radius; y <= radius; y++)
		for (x = -radius; x <= radius; x++)
			if ((x * x) + (y * y) <= (radius * radius))
				image_set_pixel (image, x0, y0, x, y, value);
}

int main (int argc, char *argv[]) {
	
	Image *image;
	image = image_new (2048, 2048);
	image_fill (image, 0);

	draw_circle (image,  256,  256, 256,  50); // Draw Label 1
	draw_circle (image,  256, 1792, 256,  50); // Draw Label 2
	draw_circle (image, 1792,  256, 256,  50); // Draw Label 3
	draw_circle (image, 1792, 1792, 256,  50); // Draw Label 4
	draw_circle (image, 1024, 1024, 512, 100); // Draw Label 5

	image_save (image, "sample.intensities.pgm");
	image_free (image);

	return 0;
}
