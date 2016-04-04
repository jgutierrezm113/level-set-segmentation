#Set test name
name="cudaTest"

cd ../../

#Create inputs
cd inputs/synthetic/
make clean
make all
make run
cd ../../

#Remove previous results
rm -rf results/$name/
mkdir -p results/$name/

#Set input files
 image="inputs/synthetic/sample.intensities.pgm"
labels="inputs/synthetic/sample.label.pgm"
params="inputs/synthetic/sample.params"

#Running tests
cd cuda
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE 16/' lib/config.h
make clean
make all
cd ../
cuda/lss --image $image --labels $labels --params $params > results/$name/$name.log
mv result.ppm results/$name/$name.ppm

cd scripts/test/
