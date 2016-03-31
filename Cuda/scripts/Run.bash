#!/bin/bash

#Calculate all results for scripts
echo "Running LSS cuda implementation. Please make sure there's a cuda device in the node."

cd ..
rm -rf results/Cuda-run/
mkdir -p results/Cuda-run/

./lss \
--image inputs/sample/sample.intensities.pgm \
--labels inputs/sample/sample.label.pgm \
--params inputs/sample/sample.params \
>>  results/Cuda-run/output.log

mv result.ppm  results/Cuda-run/output.ppm

