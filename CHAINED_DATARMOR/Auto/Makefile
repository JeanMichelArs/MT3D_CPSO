# To generate Files : > make mkfile
# To export files (Conf and auto) in rundir > Make cpfile 
# To remove netcdf files from rundir : > make rundirclean
# To submit job : > make run
# To watch output : make watch
#
# To make all of this : > make all
SHELL := /bin/bash


# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/Norme_Safe_4_JOBS/
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config/

# ---------------------------------------------------------------------
.PHONY: cpfile mkfile rundirclean run watch all

# create automatic scripts .pbs and .py for chained jobs
mkfile: 
	./mk_files.sh

# copy automatically generated files into RUN_DIR
cpfile:
	cp -f -t ${RUN_DIR} run_[0-9]*.pbs mt3d_[0-9]*.py
	cp -Rf ${CONF_DIR}/* ${RUN_DIR}

# Ensure there is no netcdf file in Run_dir
rundirclean:	
	rm ${RUN_DIR}/*.nc

# Submitt First job
run:
	cd ${RUN_DIR}; \
	qsub run_1.pbs

# check whether jobs are running and display output
watch:
	qstat -u ${USER}
	tail -f ${RUN_DIR}/output_1

# ----------------------------------------------------------------------------
# make files, copy files in run_dir, remove all netcdf files in rundir
# submit first job, watch output 
all : mkfile cpfile rundirclean run watch

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py
 
