#!/bin/bash

# Automatic copy of CPSO files for multiple runs with same configuration
# Purpose is to avoid CPSO premature convergence and enhance model space 
# exploration
#  
# > ./multi_cp_files.sh

source /usr/share/Modules/3.2.10/init/bash

# ----------------------------------------------------------------------------
# controlled by Makefile

n_runs=n_runs_xxx
RUN_DIR=RUN_DIR_XXX
CONF_DIR=CONF_DIR_XXX
MACKIE=MACKIE_XXX
#------------------------------------------------------------------------------
# local variable

if [ ! -d $RUN_DIR ]; then
  mkdir $RUN_DIR
fi

for i in $(seq 1 $n_runs)
do
  # ---> create mulitple directories
  if [ ! -d $RUN_DIR/run_$i ]; then 
    echo "creating $RUN_DIR/run_$i"
    mkdir $RUN_DIR/run_$i 
  fi
  # ---> copy configuration files, mackie, *.pbs and *.py
  echo "cp -f -t ${RUN_DIR}/run_$i/ run_[0-9]*.pbs mt3d_[0-9]*.py"
  echo " cp -Rf ${CONF_DIR}/* ${RUN_DIR}/run_$i"
  echo "cp -f ${MACKIE} ${RUN_DIR}/run_$i"
  cp -f -t ${RUN_DIR}/run_$i/ run_[0-9]*.pbs mt3d_[0-9]*.py
  cp -Rf ${CONF_DIR}/* ${RUN_DIR}/run_$i
  cp -f ${MACKIE} ${RUN_DIR}/run_$i
done
