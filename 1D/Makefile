# JCollin 2020
# --------------------------------------------------------------------
SHELL := /bin/bash

# RUN_DIR as to be the same than in mk_files.sh
RUN_DIR='/postproc/COLLIN/MTD3/1D_MCM_ana_8param'

.PHONY: merge 
merge:
	cd ${RUN_DIR}; \
	ncecat mcm_exploration*.nc -O merged_mcm.nc

# ---------------------------------------------------------------------------
.PHONY: clean_outdir

# clean local Auto directory
clean:
	cd ${RUN_DIR}; \
	rm -f mcm_exploration*.nc
	
 
