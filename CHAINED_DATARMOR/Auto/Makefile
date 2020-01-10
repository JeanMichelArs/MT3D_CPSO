# To generate Files : > make mkfile
# To export files (Conf and auto) in rundir > Make cpfile 
# To remove netcdf files from rundir : > make rundirclean
# To submit job : > make run
# To watch output : make watch
#
# To make all of this : > make all
SHELL := /bin/bash


# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/Mackie_4_JOBS/
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config/
MACKIE=/home2/datahome/jcollin/MT3D_CPSO/Example/mackie3d.so
# ---------------------------------------------------------------------
.PHONY: cpfile mkfile rundirclean run watch all

# create automatic scripts .pbs and .py for chained jobs
mkfile: 
	./mk_files.sh
	@echo ""

# copy automatically generated files into RUN_DIR
cpfile:
	@echo "copy auto files in ${RUN_DIR}"
	cp -f -t ${RUN_DIR} run_[0-9]*.pbs mt3d_[0-9]*.py 
	cp -Rf ${CONF_DIR}/* ${RUN_DIR}
	@echo "copy mackie3d in ${RUN_DIR}"
	cp -f ${MACKIE} ${RUN_DIR}/ 
	@echo ""

# Ensure there is no netcdf file in Run_dir
rundirclean:
	@echo "Removing netcdf file in ${RUN_DIR}"	
	@if [ -f ${RUN_DIR}/*.nc ]; then rm ${RUN_DIR}/*.nc; fi
	@echo ""

# Submitt First job
run:
	@echo "Submitting first job"
	cd ${RUN_DIR}; \
	qsub run_1.pbs
	@echo ""

# check whether jobs are running and display output
watch:
	qstat -u ${USER}
	@if [ -f ${RUN_DIR}/output_1 ]; then rm ${RUN_DIR}/output_1; fi
	@touch ${RUN_DIR}/output_1
	tail -f ${RUN_DIR}/output_1

# ----------------------------------------------------------------------------
# make files, copy files in run_dir, remove all netcdf files in rundir
# submit first job, watch output 
all : mkfile cpfile rundirclean run watch

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py \
	rm gen_mt3d.pbs gen_mt3d.py
 
