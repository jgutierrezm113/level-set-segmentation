#Set test name
name="coins"

cd ..

#compile cuda code
cd cuda
make clean
make all
cd ..

#compile openMP code
cd omp
make clean
make all
cd ..
 
#Remove previous results
rm -rf results/$name/
mkdir -p results/$name/seq
mkdir -p results/$name/omp
mkdir -p results/$name/cuda

#Set input files
 image="inputs/$name/$name.intensities.pgm"
labels="inputs/$name/$name.label.pgm"
params="inputs/$name/$name.params"

#Running tests

#Error using Nested Parallelism (issue initializing array)
#export OMP_NESTED=TRUE

echo "Running Sequential and OMP (32 thread) for evolution curve"
for i in 01 20 40 60  
do 
	export OMP_NUM_THREADS=1
	echo "NOTE: Using 1 thread (sequential)" > results/$name/seq/$i.log
	omp/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/seq/$i.log 
	mv result.ppm results/$name/seq/$i.ppm 
	export OMP_NUM_THREADS=32
	echo "NOTE: Using 32 threads (max allowed)" > results/$name/omp/$i.log
	omp/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/omp/$i.log
	mv result.ppm results/$name/omp/$i.ppm
done

echo "Running Cuda for evolution curve"
for i in 1 2 3 4
do
	echo "NOTE: Using block size 16x16 (default)" > results/$name/cuda/$i.log
	cuda/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/cuda/$i.log
	mv result.ppm results/$name/cuda/$i.ppm
done

cd scripts/
