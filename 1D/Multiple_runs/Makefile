# JCollin 2020
# --------------------------------------------------------------------

SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR='/postproc/COLLIN/MTD3/MCM_8nz_cst_Error'
n_start=1
n_end=2
j=${n_start}

# ----------------------------------------------------------------------------
#  Multiple runs of 1d cpso

.PHONY: multi_run multi_ana

# ---> analytical forward modelling
multi_ana:
	for i in $(shell seq ${n_start} ${n_end}); do \
	  (python  MT1D_analytic_CPSO.py $$i &> log.$$i &); \
	done

# ---> mackie 1D run 
multi_run: 
	for i in $(shell seq ${n_start} ${n_end}); do \
	  (python MT1D_CPSO_chained.py $$i &> log.$$i &); \
	done

# ---> watch job progress, does not work
.PHONY: watch

watch:
	tail -f log.0

# ----------------------------------------------------------------------------
#  ${RUN_DIR}/merged.nc must be removed first
.PHONY: merge merge50
merge:
	cd ${RUN_DIR}; \
	ncecat *exploration*nc -O merged.nc

# ----------------------------------------------------------------------------


.PHONY: clean

# clean local Auto directory
clean:
	rm -f log.*
	
 
