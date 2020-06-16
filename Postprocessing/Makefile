#!/bin/bash 
# JCOLLIN 03-2020
# Postprocessing for CPSO
# 
# Feature:
# - import: import files from remote serveur to local for postprocessing
# - p_best : create a netcdf file with pbest(it) best mysift
#
# activate your conda environment
#
# generic files may be edited (file_name_gen.py) 
# ----------------------------------------------------------------------------

SHELL := /bin/bash

.PHONY: p_best import clean

# ----------------------------------------------------------------------------

# ---> import:
#
#  import files from remote serveur to local for postprocessing
#  TODO : loop importation over a list of files
#
local_path=/postproc/COLLIN/MTD3
user=${USER}
host=datarmor

# user settings
run=bolivia_24param_mbest_zoom
run_dir=${local_path}/${run}
#datar_path=/home2/scratch/jcollin/MT3D_CPSO/${run}
datar_path=/home2/datawork/jcollin/MT3D_CPSO/${run}
import_file=merged.nc

import:
	if [ ! -d ${run_dir} ] ; then mkdir ${run_dir} ; fi
	scp ${user}@${host}:/${datar_path}/${import_file} ${run_dir}/

# ---> p_best:
#
#   create a netcdf file with pbest(it) best mysift
#   niter_zoom = allows to compute mysfit over niter_zoom last iteration
#   also provides the total number of particles statistically relevant
#   for marginal pdf, best likehood <m>... 

RUN_DIR=${DATAWORK}/MT3D_CPSO/sensi_analysis/Bolivia_115param_015
nc_file=${RUN_DIR}/merged.nc
mod_rho=${}/home2/datahome/sflora/MT3D_CPSO_TEST/sensi_analysis/Bolivia_115param_015/MT3D_BOLIVIA_IMAGIRb.fmt
mod_param=${}/home2/datahome/sflora/MT3D_CPSO_TEST/sensi_analysis/Bolivia_115param_015/parameter_model.ini
refN=622901.875
refE=7513822.0
ang=0.0

p_best:  
	cp -f pbest_gen.py pbest.py 
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" pbest.py
	sed -i "s:nc_file_xxx:"${nc_file}":g" pbest.py
	python pbest.py |& tee pbest.log 
	
# ----------------------------------------------------------------------------

j_plot:  
	cp -f Jointplot_gen.py Jointplot.py 
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" Jointplot.py
	sed -i "s:nc_file_xxx:"${nc_file}":g" Jointplot.py
	sed -i "s:mod_rho_xxx:${mod_rho}:g" Jointplot.py
	sed -i "s:mod_param_xxx:"${mod_param}":g" Jointplot.py
	sed -i "s:refN_xxx:"${refN}":g" Jointplot.py
	sed -i "s:refE_xxx:"${refE}":g" Jointplot.py
	sed -i "s:ang_xxx:"${ang}":g" Jointplot.py
	qsub j_plot.pbs
	
# ----------------------------------------------------------------------------

clean:
	rm *.log *.py
