# environment
CUDA_VERSION := cuda-7.5
SM := 35
OS := $(shell uname -s 2>/dev/null | tr a-z A-Z)

# compilers
ifeq ($(OS), DARWIN)
	CUDA_PATH ?= /Developer/NVIDIA/$(CUDA_VERSION)
else
	CUDA_PATH ?= /usr/local/$(CUDA_VERSION)
endif
GCC := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# libraries
CUDA_LIB_PATH := $(CUDA_PATH)/lib

# flags
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart
ifeq ($(OS), DARWIN)
	CCFLAGS := -stdlib=libstdc++
else
	CCFLAGS :=
endif
NVCCFLAGS :=

all: build

build: lss

lss: dlink.o main.o lss.o config.o utility.o
	@# $(GCC) $^ -o $@ -L$(CUDA_LIB_PATH) $(LIB_FLAGS) $(CCFLAGS)
	$(NVCC) $^ -o $@ $(GENCODE_FLAGS) -link

dlink.o: lss.o config.o
	$(NVCC) $^ -o $@ $(GENCODE_FLAGS) -dlink

main.o: main.cpp
	$(GCC) -c $< -o $@
lss.o: lss.cu
	$(NVCC) -dc $< -o $@ $(GENCODE_FLAGS) $(NVCCFLAGS) 
config.o: config.cu
	$(NVCC) -dc $< -o $@ $(GENCODE_FLAGS) $(NVCCFLAGS)

utility.o: utility.cpp
	$(GCC) -c $< -o $@ $(CCFLAGS)

run: build
	./lss

clean:
	rm -f *.o
	rm -f lss
