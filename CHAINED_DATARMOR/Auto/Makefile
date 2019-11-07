# To copy Auto and Conf files into Run_dir
# > Make cp_file 

SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/BASIC_FUN_2_JOBS/
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config/


# copy automatically generated files into RUN_DIR
.PHONY: cp_file
cp_file:
	cp -f -t ${RUN_DIR} run_[0-9]*.pbs mt3d_[0-9]*.py
	cp -Rf ${CONF_DIR}/* ${RUN_DIR}

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py
 
