# level-set-segmentation
Northeastern University

EECE5640 - High Performance Computing

Spring, 2016

Julian Gutierrez

The proposed project is a follow up for an ongoing research project by students at the Northeastern University Computer Architecture Research lab [1]. The idea is to enhance the implementation of the level set segmentation algorithm using parallel coding, targeting efficiency of execution in GPU’s and cluster computing. This algorithm is commonly used in image segmentation, partitioning an image into regions of interest to be able to detect specific objects in the image. The use of parallelism in this code is fundamental due to the fact that it is applied to images, where the analysis for each pixel can be done in parallel (embarrassingly parallel, mixed with a sort of pipelined type of analysis). Also, as stipulated in the next chapters, we can apply such algorithm to multiple objects and multiple images at the same time, increasing the parallelizability of the code.

The overall objective was met, which was to create an application that is efficient in execution so that it can be added as part of the NUPAR benchmark suite and help target performance on modern architectures. This was done by the implementation of a new algorithmic approach to the problem, and by implementing the code in Cuda using dynamic parallelism (nested), the performance was significantly improved. 

Note: to run full experiments please go to the experiments folder and follow the steps there.


