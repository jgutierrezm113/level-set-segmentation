#ifndef LSS_H
#define LSS_H

#include "config.h"
#include "eventTimer.h"

__global__ void initDifference(udat *differenceDev, udat *intensityDev, udat *componentDev, int size, int targetLabel);
__global__ void initPhi(udat *componentDev, dat *phiDev, int height, int width, int targetLabel);

extern __device__ udat *difference;
extern __device__ int *s1, *s2, *cnt;

__global__ void sumImage(int *s1Dev, int *s2Dev, int *cntDev, udat *differenceDev, dat *phiDev, int size);
__global__ void averageImage(float *c1Dev, float *c2Dev, int *s1Dev, int *s2Dev, int *cntDev, int n, int size);

__global__ void evolveContour(udat *intensityDev, udat *componentDev, dat *phiDev, int height, int width, int imageId, int *labelsDev, int numberOfImages, int numberOfLabels);

__global__ void switchIn(udat *differenceDev, float c1Dev, float c2Dev, signed char* phiDev, int height, int width);
__global__ void switchOut(udat *differenceDev, float c1Dev, float c2Dev, signed char* phiDev, int height, int width);

__global__ void checkStop(udat *differenceDev, float c1Dev, float c2Dev, dat* phiDev, int size);

signed char *levelSetSegment(unsigned char *intensity, unsigned char *component, int height, int width, int numberOfImages, int *labels, int numberOfLabels);

#endif
