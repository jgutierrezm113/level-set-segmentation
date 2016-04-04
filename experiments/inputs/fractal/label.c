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
 * the beginning of the image.
 **/
static void image_set_pixel (Image *image, ssize_t x, ssize_t y, unsigned char  value) {
	unsigned char *p;

	p = image->data + (y * image->width) + x;
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

static void draw_label (Image *image, int x0, int y0, int x1, int y1, unsigned char  value) {
	int x, y;
	for (y = y0; y < y1; y++)
		for (x = x0; x < x1; x++)
			image_set_pixel (image, x, y, value);
}

int main (int argc, char *argv[]) {
	
	Image *image;
	image = image_new (1600, 1600);
	image_fill (image, 0);

	draw_label (image, 760, 1550,  840, 1600,  100); // Draw Label 1

	image_save (image, "fractal.label.pgm");
	image_free (image);

	return 0;
}
