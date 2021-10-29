#!/bin/bash
for i in {1..12}
do 
	echo "run $i:"
	python3 goldilocks_vectorfield_FI_early_incremental.py 
	cp goldilocks_vectorfield_logfile_early_inc.log goldilocks_vectorfield_logfile_early_inc$i.log
	cp goldilocks_FI_logfile_early_inc.log goldilocks_FI_logfile_early_inc_$i.log

done


