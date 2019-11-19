# To copy Auto and Conf files into Run_dir
# To genrate Files : ./mk_files
# > Make cpfile 

SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/Norme_Safe_4_JOBS/
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config/


# copy automatically generated files into RUN_DIR

.PHONY: cpfile
cpfile:
	cp -f -t ${RUN_DIR} run_[0-9]*.pbs mt3d_[0-9]*.py
	cp -Rf ${CONF_DIR}/* ${RUN_DIR}

# Ensure there is no netcdf file in Run_dir
.PHONY: rundirclean
rundirclean:	
	rm ${RUN_DIR}/*.nc

.PHONY: run
run:
	cd ${RUN_DIR}; \
	qsub run_1.pbs

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py
 
