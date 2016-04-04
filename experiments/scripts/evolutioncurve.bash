#Set test name
name="evolutioncurve"

cd ..

#Create inputs
cd inputs/$name/
make clean
make all
make run
cd ../../

#Remove previous results
rm -rf results/$name/
mkdir -p results/$name/cuda
mkdir -p results/$name/omp
mkdir -p results/$name/seq

#Set input files
 image="inputs/$name/sample.intensities.pgm"
labels="inputs/$name/sample.label.pgm"
params="inputs/$name/sample.params"

#Running tests
echo "Running Cuda for different block sizes"

###################################
bs=8
echo "Compiling for block size $bs"

cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

for i in 01 06 12 18
do
	echo "Testing block size $bs" >> results/$name/cuda/$bs.log
	cuda/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/cuda/$bs.log
	mv result.ppm results/$name/cuda/$bs.$i.ppm
done

###################################
bs=16
echo "Compiling for block size $bs"

cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

for i in 01 03 06 10
do
	echo "Testing block size $bs" >> results/$name/cuda/$bs.log
	cuda/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/cuda/$bs.log
	mv result.ppm results/$name/cuda/$bs.$i.ppm
done

###################################
bs=32
echo "Compiling for block size $bs"

cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

for i in 01 02 03 05
do
	echo "Testing block size $bs" >> results/$name/cuda/$bs.log
	cuda/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/cuda/$bs.log
	mv result.ppm results/$name/cuda/$bs.$i.ppm
done

#setting cuda back to default
cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE 16/' lib/config.h
make clean
make all
cd ..

#compile openMP code
cd openMP
make clean
make all
cd ..
 
#Error using Nested Parallelism (issue initializing array)
#export OMP_NESTED=TRUE

echo "Running Sequential and OMP (32 thread) for evolution curve"
for i in 001 200 400 510  
do 
	export OMP_NUM_THREADS=1
	echo "NOTE: Using 1 thread (sequential)" > results/$name/seq/$i.log
	openMP/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/seq/$i.log 
	mv result.ppm results/$name/seq/$i.ppm 
	export OMP_NUM_THREADS=32
	echo "NOTE: Using 32 threads (max allowed)" > results/$name/omp/$i.log
	openMP/lss --image $image --labels $labels --params $params --max_reps $i >> results/$name/omp/$i.log
	mv result.ppm results/$name/omp/$i.ppm
done

cd scripts/
