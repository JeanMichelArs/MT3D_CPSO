#!/bin/bash

# Automatic copy of CPSO files for multiple runs with same configuration
# Purpose is to avoid CPSO premature convergence and enhance model space 
# exploration
#  
# > ./multi_cp_files.sh

source /usr/share/Modules/3.2.10/init/bash

# ----------------------------------------------------------------------------
# controlled by Makefile

n_jobs=2
RUN_DIR=/home2/scratch/jcollin/MT3D_CPSO/xstart_rand_Mackie_33param
CONF_DIR=/home2/datahome/jcollin/MT3D_CPSO/Config
MACKIE=/home2/datahome/jcollin/MT3D_CPSO/mackie3d.so
#------------------------------------------------------------------------------
# local variable

if [ ! -d $RUN_DIR ]; then
  mkdir $RUN_DIR
fi

for i in $(seq 1 $n_jobs)
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
