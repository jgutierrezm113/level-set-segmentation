#Set test name
name="optimizations"

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
mkdir -p results/$name/cudaNoOpt
mkdir -p results/$name/cudaOpt

#Note
echo "**********************************************"
echo "* PLEASE NOTE CUDA IS MOST OPTIMIZED VERSION *"
echo "**********************************************"
sleep 5

#Set input files
 image="inputs/$name/sample.intensities.pgm"
labels="inputs/$name/sample.label.pgm"
params="inputs/$name/sample.params"

#Running tests
echo "Running Cuda for different optimizations"

###################################
echo "Compiling for tile size 32"

bs=16
cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

bs=32
cd cudaOpt
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

cd cudaNoOpt
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make clean 
make all
cd ../

echo "Testing with tile size $bs" >> results/$name/cuda/result.log
echo "Testing with tile size $bs" >> results/$name/cudaOpt/result.log
echo "Testing with tile size $bs" >> results/$name/cudaNoOpt/result.log

       cuda/lss --image $image --labels $labels --params $params >> results/$name/cuda/result.log
nvprof cuda/lss --image $image --labels $labels --params $params &> results/$name/cuda/nvprof.log

mv result.ppm results/$name/cuda/

       cudaOpt/lss --image $image --labels $labels --params $params >> results/$name/cudaOpt/result.log
nvprof cudaOpt/lss --image $image --labels $labels --params $params &> results/$name/cudaOpt/nvprof.log

mv result.ppm results/$name/cudaOpt/

       cudaNoOpt/lss --image $image --labels $labels --params $params >> results/$name/cudaNoOpt/result.log
nvprof cudaNoOpt/lss --image $image --labels $labels --params $params &> results/$name/cudaNoOpt/nvprof.log

mv result.ppm results/$name/cudaNoOpt/

cd scripts/
