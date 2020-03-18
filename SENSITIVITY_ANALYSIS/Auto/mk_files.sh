#!/bin/bash

# Automatic generation of chained jobs to break wall time
# set number of jobs, number of iteration per job
# > ./mk_files.sh

source /usr/share/Modules/3.2.10/init/bash

# -------------------------------------------------------------
# USER PARAMETERS
# -------------------------------------------------------------
# - directory to run simulation
# - total number of chained jobs
# - total number of iteration (sum of max_iter per job)
# - max_iter must be a multiple of n_jobs

RUN_DIR="${SCRATCH}/MT3D_CPSO/bolivia_24param_mbest_zoom"

n_jobs=20
max_iter=1000

# Former parameter found in input file
# ** JARS add new model
# - parameter_model: model structure to represent geophysical structure
# this technique allows to minimize the total number of parameter
# - model_rho: center of parameter space to be explored
# may be solution from deterministic method or another cpso run...
parameter_model="parameter_model.ini"
model_rho="update_model.ini"
ref_North=7513822.0
ref_East=622901.875
angle_grid=0.0
MTsounds_positions="MTcolorada_sounding.pos"
popsize=56
# ----> added method and window for parameter research space
# - init_xstart: a priori law on parameters 
# - cst_upper, cst_lower window param \in [-lower, upper]^nparam
init_xstart="xi_rand_uniform"
cst_lower=0.3
cst_upper=0.3



# Python and pbs scripts
# ** JARS Change files 
MT3D_py="../MT3D_CPSO_sensi_analysis.py"
MT3D_pbs="../mt3d_cpso_sensi_analysis.pbs"

# -------------------------------------------------------------
# END OF USER PARAMETERS
# -------------------------------------------------------------
# ** JARS do not modify this
gen_py="gen_mt3d.py"
gen_pbs="gen_mt3d.pbs"

cp -f ${MT3D_py} ${gen_py}
cp -f ${MT3D_pbs} ${gen_pbs}
# ---------------------------------------------------------------
echo "Total number of chained jobs:" $n_jobs
echo "Generic PBS script" $gen_pbs
echo "Generic mt3d python script:" $gen_py
# -------------------------------------------------------------

if [ ! -f "$RUN_DIR" ]; then
    echo "create directory $RUN_DIR"
    mkdir $RUN_DIR
fi

# ---
echo "Cleaning local directory from old files"
rm run_[0-9]*.pbs
rm mt3d_[0-9]*.py

# Building sources from gen_script and gen_run 
i_job=1
while [ $i_job -le $n_jobs ]
do
  echo ""
  echo "creating scripts " $i_job "out of " $n_jobs
  # copy from genereic files
  cp ${gen_pbs} run_${i_job}.pbs
  cp ${gen_py} mt3d_${i_job}.py

  # ---> modify i_job, n_jobs, max_iter in python
  sed -i "s/i_job_xxx/${i_job}/g" mt3d_${i_job}.py
  sed -i "s/n_jobs_xxx/${n_jobs}/g" mt3d_${i_job}.py
  sed -i "s/max_iter_xxx/${max_iter}/g" mt3d_${i_job}.py

  #----> input file parameters
  # ** JARS could add something here but easier to write
  # in MT3D_CPSO_sensi_analysis.py equivalent
  sed -i "s/parameter_model.ini_xxx/'${parameter_model}'/g" mt3d_${i_job}.py
  sed -i "s/model_rho_xxx/'${model_rho}'/g" mt3d_${i_job}.py
  sed -i "s/refmty_xxx/${ref_East}/g" mt3d_${i_job}.py
  sed -i "s/refmtx_xxx/${ref_North}/g" mt3d_${i_job}.py
  sed -i "s/angm_xxx/${angle_grid}/g" mt3d_${i_job}.py
  sed -i "s/MTsoundingpos_xxx/'${MTsounds_positions}'/g" mt3d_${i_job}.py
  sed -i "s/popsize_xxx/${popsize}/g" mt3d_${i_job}.py
  
  sed -i "s/init_xstart_xxx/'${init_xstart}'/g" mt3d_${i_job}.py
  sed -i "s/cst_lower_xxx/${cst_lower}/g" mt3d_${i_job}.py
  sed -i "s/cst_upper_xxx/${cst_upper}/g" mt3d_${i_job}.py

  # ---> modify rundir, i_jobs 
#  sed -i "s:RUN_DIR_XXX:${RUN_DIR}:g" run_${i_job}.pbs
  sed -i "s/mt3d_xxx.py/mt3d_${i_job}.py/g" run_${i_job}.pbs
  sed -i "s/output_xxx/output_${i_job}/g" run_${i_job}.pbs
  sed -i "s/mt3d_xxx.pbs/run_$(($i_job+1)).pbs/g" run_${i_job}.pbs
  ((i_job++))
done
# modify last run_N_jobs.pbs in ordrer not to submit non existing job
sed -i "s/qsub run_${i_job}.pbs/echo 'End of Run'/g" run_${n_jobs}.pbs
