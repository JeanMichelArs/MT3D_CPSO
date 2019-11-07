#!/bin/bash

source /usr/share/Modules/3.2.10/init/bash

# -------------------------------------------------------------
# USER PARAMETERS
# total number of chained jobs
# total number of iteration (sum of max_iter per job)
# SRC 
# max_iter must be a multiple of n_jobs
RUN_DIR="/home2/scratch/jcollin/MT3D_CPSO/BASIC_FUN_2_JOBS/"

n_jobs=2
max_iter=50

MT3D_py="../MT3D_CPSO_datar_chained.py"
MT3D_pbs="../mt3d_cpso_chained.pbs"

# END OF USER PARAMETERS
# -------------------------------------------------------------
gen_py="gen_mt3d.py"
gen_pbs="gen_mt3d.pbs"

cp -f ${MT3D_py} ${gen_py}
cp -f ${MT3D_pbs} ${gen_pbs}
# ---------------------------------------------------------------
echo "Total number of chained jobs:" $n_jobs
echo "Generic PBS script" $gen_pbs
echo "Generic mt3d python script:" $gen_py
# -------------------------------------------------------------

# Building sources from gen_script and gen_run 
i_job=1
while [ $i_job -le $n_jobs ]
do
  echo ""
  echo "creating scripts " $i_job "out of " $n_jobs
  # copy from gen_files
  cp ${gen_pbs} run_${i_job}.pbs
  cp ${gen_py} mt3d_${i_job}.py
  # ---> modify i_job, n_jobs in python
  sed -i "s/i_job_xxx/${i_job}/g" mt3d_${i_job}.py
  sed -i "s/n_jobs_xxx/${n_jobs}/g" mt3d_${i_job}.py
  sed -i "s/max_iter_xxx/${max_iter}/g" mt3d_${i_job}.py
  # change in run: script to be executed, next qsub, log and err files
  # in script just index to be printed
#  sed -i "s/RUN_DIR=RUN_DIR_XXX/RUN_DIR=${RUN_DIR}/g" run_${i_job}.pbs
  sed -i "s:RUN_DIR_XXX:${RUN_DIR}/g" run_${i_job}.pbs
  sed -i "s/mt3d_xxx.py/mt3d_${i_job}.py/g" run_${i_job}.pbs
  sed -i "s/output_xxx/output_${i_job}/g" run_${i_job}.pbs
  sed -i "s/mt3d_xxx.pbs/run_$(($i_job+1)).pbs/g" run_${i_job}.pbs
  ((i_job++))
done
# modify last run_N_jobs.pbs in ordrer not to submit non existing job
sed -i "s/qsub run_${i_job}.pbs/echo 'End of Run'/g" run_${n_jobs}.pbs
#sed -i "s/qsub gen_run_next.pbs/echo 'End of chained Run' /" run_${i_job}.pbs
