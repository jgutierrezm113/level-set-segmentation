#!/bin/sh 

#BSUB -J JGM-OMP 
#BSUB -o output.log 
#BSUB -e err.log
#BSUB -n 4 
#BSUB -q ht-10g 
#BSUB cwd /home/gutierrez.jul/HPC/project/level-set-segmentation/OpenMP 

work=/home/gutierrez.jul/HPC/project/level-set-segmenatation/OpenMP

cd $work 
rm -rf results/discovery-omp
mkdir results/discovery-omp/

for i in 1 2 4 8 16 32 64 128 
do 
	export OMP_NUM_THREADS=$i 
	./lss --image inputs/sample/sample.intensities.pgm --labels inputs/sample/sample.label.pgm --params inputs/sample/sample.params
	mv result.ppm results/discovery-omp/result.$i.ppm 
done

