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


# ---> p_best
# create a netcdf file with pbest(it) best mysift
# niter_zoom = allows to compute mysfit over niter_zoom last iteration
# also provides the total number of particles statistically relevant
# for marginal pdf, best likehood <m>... 

RUN_DIR=${DATAWORK}/MT3D_CPSO/bolivia_24param
#RUN_DIR=${SCRATCH}/MT3D_CPSO/bolivia_24param
nc_file=${RUN_DIR}/merged.nc
#nc_file=${RUN_DIR}/run_1/cpso_mackie_33_param.nc
mod_rho=${}
mod_param=${}
refN=622901.875
refE=7513822.0
ang=0

p_best:  
	cp -f pbest_gen.py pbest.py 
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" pbest.py
	sed -i "s:nc_file_xxx:"${nc_file}":g" pbest.py
	python pbest.py |& tee pbest.log 
	
# ----------------------------------------------------------------------------

j_plot:  
	cp -f jointplot_gen.py jointplot.py 
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" jointplot.py
	sed -i "s:nc_file_xxx:"${nc_file}":g" jointplot.py
	sed -i "s:mod_rho_xxx:${mod_rho}:g" jointplot.py
	sed -i "s:mod_param_xxx:"${mod_param}":g" jointplot.py
	sed -i "s:refN_xxx:"${refN}":g" jointplot.py
	sed -i "s:refE_xxx:"${refE}":g" jointplot.py
	sed -i "s:ang_xxx:"${ang}":g" jointplot.py
	python jointplot.py |& tee jointplot.log 
	
# ----------------------------------------------------------------------------

clean:
	rm *.log *.py
