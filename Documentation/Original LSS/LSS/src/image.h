/* a simple image class */

#ifndef IMAGE_H
#define IMAGE_H

#include <cstring>

template <class T>
class Image {
 public:
  /* create an image */
	Image(const int width, const int height, const bool init = true);

  /* delete an image */
  ~Image();

  /* init an image */
  void init(const T &val);

  /* copy an image */
  Image<T> *copy() const;
  
  /* get the width of an image. */
  int width() const { return w; }
  
  /* get the height of an image. */
  int height() const { return h; }
  
  /* image data. */
  T *data;
  
  /* row pointers. */
  T **access;
  
 private:
  int w, h;
};

/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])
  
/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])

template <class T>
Image<T>::Image(const int width, const int height, const bool init)
{
  w = width;
  h = height;
  data = new T[w * h];  // allocate space for image data
  access = new T*[h];   // allocate space for row pointers
  
  // initialize row pointers
  for (int i = 0; i < h; i++)
    access[i] = data + (i * w);
  
  if (init)
    memset(data, 0, w * h * sizeof(T));
}

template <class T>
Image<T>::~Image()
{
  delete [] data; 
  delete [] access;
}

template <class T>
void Image<T>::init(const T &val)
{
  T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, w-1, h-1);
  while (ptr <= end)
    *ptr++ = val;
}


template <class T>
Image<T> *Image<T>::copy() const
{
	Image<T> *im = new Image<T>(w, h, false);
  memcpy(im->data, data, w * h * sizeof(T));
  return im;
}


#endif
  
