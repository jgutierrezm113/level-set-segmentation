#Set test name
name="blocksize"

cd ..

#Create inputs
cd inputs/$name/
make clean
make all
make run
cd ../../

#Remove previous results
rm -rf results/$name/
mkdir -p results/$name/

#Set input files
 image="inputs/$name/sample.intensities.pgm"
labels="inputs/$name/sample.label.pgm"
params="inputs/$name/sample.params"

#Running tests
echo "Running Cuda for different block sizes"
for bs in 2 4 8 16 32
do
	echo "Compiling for block size $bs"
	cd cuda
	sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
	make clean 
	make all
	cd ../
	echo "Testing block size $bs" > results/$name/$bs.log
	cuda/lss --image $image --labels $labels --params $params >> results/$name/$bs.log
	mv result.ppm results/$name/$bs.ppm
	nvprof cuda/lss --image $image --labels $labels --params $params &> results/$name/$bs.nvprof.log
done

#setting cuda back to default
cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE 16/' lib/config.h
make clean
make all
cd ..

#removing nvprof output
rm result.ppm

cd scripts/

#Note: Last value is 16 so to leave the block size to the default vale
