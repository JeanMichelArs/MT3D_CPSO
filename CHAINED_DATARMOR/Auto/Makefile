# JCollin 2020
#----> MULTIPLE CPSO RUN
# To clean run directory : > make multi_clean 
# To generate cpso chained files : > make multi_mkfile
# To duplicate data for multiple runs : > make multi_cpfile
# To submit every first job : > make multi_run 

# ---> SINGLE CPSO RUN
# To generate Files : > make mkfile
# To export files (Conf and auto) in rundir > Make cpfile 
# To remove netcdf files from rundir : > make rundirclean
# To submit job : > make run
# To watch output : make watch
#
# To make all of this : > make all
SHELL := /bin/bash


# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/multiple_cpso/
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config/
MACKIE=/home2/datahome/jcollin/MT3D_CPSO/mackie3d.so
n_runs=20
# ---------------------------------------------------------------------
# Single run of cpso
# Ideal for tests or estimating number of iteration for convergence
# However there is a risk of premature convergence in local minima
# for multiple runs of cpso see below

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
# Runs multiple times the same problem with a different initial model
# 

.PHONY: multi_mkfile multi_cpfile multi_clean multi_run multi_all multi_watch

# creates cpso files for chained jobs
multi_mkfile:
	./mk_files.sh
	@echo ""

# duplicates run in separate folders
multi_cpfile:
	cp -f multi_cp_files_gen.sh multi_cp_files.sh
	sed -i "s:n_runs_xxx:${n_runs}:g" multi_cp_files.sh
	sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" multi_cp_files.sh
	sed -i "s:CONF_DIR_XXX:${CONF_DIR}:g" multi_cp_files.sh
	sed -i "s:MACKIE_XXX:${MACKIE}:g" multi_cp_files.sh
	./multi_cp_files.sh

# submit all run
multi_run:
	for i in $(shell seq 1 ${n_runs}); do \
            cd ${RUN_DIR}/run_$$i; qsub run_1.pbs ;\
        done

# watch fist job
multi_watch:
	qstat -u ${USER}
	@if [ -f ${RUN_DIR}/run_1/output_1 ]; then rm ${RUN_DIR}/run_1/output_1; fi
	@touch ${RUN_DIR}/run_1/output_1
	tail -f ${RUN_DIR}/run_1/output_1

# clean directories
multi_clean:
	rm -rf ${RUN_DIR}/*

mutli_qdel:
	id=7690005; \ 
	for i in $(shell seq 1 ${n_runs}); do \
	  qdel ;\
        done

## Full 
multi_all : multi_clean multi_mkfile multi_cpfile multi_run multi_clean

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py \
	rm gen_mt3d.pbs gen_mt3d.py
 
