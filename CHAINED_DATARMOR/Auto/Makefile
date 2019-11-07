SHELL := /bin/bash
RUN_DIR="/home2/scratch/jcollin/MT3D_CPSO/BASIC_FUN_2_JOBS/"


.PHONY: copy_file
copy_file:
	cp -f -t $$RUN_DIR run_[0-9]*.pbs mt3d_[0-9]*.py

# ----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f *.out *.o[0-9]* *.log *.err run_[0-9]*.pbs mt3d_[0-9]*.py
 
