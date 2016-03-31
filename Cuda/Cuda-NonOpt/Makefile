# environment
CUDA_VERSION := cuda-6.0
SM := 35

# compilers
# For discovery cluster
CUDA_PATH ?= /shared/apps/cuda6.0

GCC := g++
NVCC := $(CUDA_PATH)/bin/nvcc
MPI  = mpiCC

# libraries
CUDA_LIB_PATH := $(CUDA_PATH)/lib

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart
ifeq ($(OS), DARWIN)
	CCFLAGS := -stdlib=libstdc++
else
	CCFLAGS :=
endif
NVCCFLAGS :=
GccFLAGS = -fopenmp 
MPIFLAGS = -Wno-deprecated

debug: GccFLAGS += -DDEBUG -g -Wall
debug: MPIFLAGS += -DDEBUG
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = lss

all: build

build: $(TARGET)

lss: lib/dlink.o lib/main.o lib/lss.o lib/imghandler.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -link

lib/dlink.o: lib/lss.o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

lib/main.o: lib/main.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
lib/lss.o: lib/lss.cu 
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) $(NVCCFLAGS) 
	
lib/imghandler.o: lib/imghandler.cpp
	$(GCC) $(GccFLAGS) -c $< -o $@ $(CCFLAGS)
	
clean:
	$(RM) $(TARGET) *.o lib/*.o *.tar* *.core*
