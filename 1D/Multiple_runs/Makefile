# JCollin 2020
# --------------------------------------------------------------------

SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR='/postproc/COLLIN/MTD3/Bolivia_1D_8param'
n_runs=2

# ----------------------------------------------------------------------------
#  Multiple runs of 1d cpso

.PHONY: multi_run

multi_run: 
	for i in $(shell seq 1 ${n_runs}); do \
	  (python MT1D_CPSO_chained.py $$i &> log.$$i &); \
	done

# ----------------------------------------------------------------------------
merge:
	cd ${RUN_DIR}; \
	ncecat *nc -O merged.nc

# ---------------------------------------------------------------------------

#source /usr/share/Modules/3.2.10/init/bash; \
module load nco/4.7.1_conda; \
ncecat *nc -O merged.nc  	
# ----------------------------------------------------------------------------
.PHONY: clean

# clean local Auto directory
clean:
	rm -f log.*
	
 
