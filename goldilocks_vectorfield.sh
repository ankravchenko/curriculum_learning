#!/bin/bash
for i in {1..12}
do 
	echo "run $i:"
	python3 goldilocks_vectorfield.py 
	cp goldilocks_vectorfield_logfile.log goldilocks_vectorfield_logfile_e4_$i.log

done


