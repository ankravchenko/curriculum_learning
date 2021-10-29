#!/bin/bash
for i in {1..12}
do 
	echo "run $i:"
	python3 goldilocks_vectorfield_FI_deep_full.py 
	cp goldilocks_vectorfield_logfile_deep_full_early.log goldilocks_vectorfield_logfile_deep_full_early$i.log
	cp goldilocks_FI_logfile_deep_full_early.log goldilocks_FI_logfile_deep_full_early$i.log


done


