# JCollin 2020
# --------------------------------------------------------------------

SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR='/postproc/COLLIN/MTD3/Bolivia_1D_8param'
n_start=151
n_end=200
j=${n_start}

# ----------------------------------------------------------------------------
#  Multiple runs of 1d cpso

.PHONY: multi_run

multi_run: 
	for i in $(shell seq ${n_start} ${n_end}); do \
	  (python MT1D_CPSO_chained.py $$i &> log.$$i &); \
	done

.PHONY: watch

watch:
	tail -f log.$$j

# ----------------------------------------------------------------------------
#  ${RUN_DIR}/merged.nc must be removed first
.PHONY: merge merge50
merge:
	cd ${RUN_DIR}; \
	ncecat test*nc -O merged.nc

merge50:
	cd ${RUN_DIR}; \
	ncecat test[0-9].nc test[0-4][0-9].nc -O merged.nc


# ---------------------------------------------------------------------------

#source /usr/share/Modules/3.2.10/init/bash; \
module load nco/4.7.1_conda; \
ncecat *nc -O merged.nc  	
# ----------------------------------------------------------------------------
.PHONY: clean

# clean local Auto directory
clean:
	rm -f log.*
	
 
