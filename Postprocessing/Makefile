#!/bin/bash 
# JCOLLIN 03-2020
# Postprocessing for CPSO
# 
# Feature:
# p_best : create a netcdf file with pbest(it) best mysift
#
# activate your conda environment
#
# generic files may be edited (file_name_gen.py) 
# ----------------------------------------------------------------------------

SHELL := /bin/bash

.PHONY: p_best clean

RUN_DIR=${DATAWORK}/MT3D_CPSO/multiple_cpso
nc_file=${RUN_DIR}/merged.nc

p_best:  
	cp -f pbest_gen.py pbest.py 
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" pbest.py
	sed -i "s:nc_file_xxx:"${nc_file}":g" pbest.py
	python pbest.py |& tee pbest.log 
	
# ----------------------------------------------------------------------------

clean:
	rm *.log *.py
