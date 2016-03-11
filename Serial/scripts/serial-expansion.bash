#!/bin/bash

#Calculate all results for scripts
echo "Running Expansion sequence, this will take time."

cd ..
mkdir serial-expansion/

for ((i=1; i <= 501; i=i+50)); do
	echo "Running Serial expansion for max iteration of $i."
	./lss \
	--image inputs/sample/sample.intensities.pgm \
	--labels inputs/sample/sample.label.pgm \
	--params inputs/sample/sample.params \
	--max_reps $i >>  serial-expansion/output.log

	mv result.ppm  serial-expansion/output.$i\.ppm

done

rm -rf results/serial-expansion

mv  serial-expansion results/
