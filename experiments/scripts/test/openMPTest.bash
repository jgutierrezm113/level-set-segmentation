#Set test name
name="openMPTest"

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
cd openMP
make clean
make all
cd ../
openMP/lss --image $image --labels $labels --params $params > results/$name/$name.log
mv result.ppm results/$name/$name.ppm

cd scripts/test/
