"""
------------------------------------------------------------------------------
plot_pdf.py
jeudi 8 octobre 2020, 12:01:11 (UTC+0200)

script to plot marginal laws, parameter space exploration, synthetic model,
<m> and mbest

------------------------------------------------------------------------------
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

sys.path.append('../')
from cpso_pp import plot_pdfm

# ----------------------------------------------------------------------------
nc_path = '/postproc/COLLIN/MTD3/MCM_16nz_cst_Error/Analysis'
ncfile = nc_path + '/compressed_pdf_m_16.nc'
save_dir = nc_path
figname = save_dir + '/pdfm_test'

# ---> get data
nc = Dataset(ncfile, 'r')
m_synth = nc.variables['m_synth'][:]
m_weight = nc.variables['m_weight'][:]
n_bin = nc.variables['n_bin'][:]
pdf_m = nc.variables['pdf_m'][:]
std_weight = nc.variables['std_weight'][:]
x_bin = nc.variables['x_bin'][:]
nc.close()

# ---> plot marginal laws

plot_pdfm(gen_name=figname, pdf_m=pdf_m, x_bin=x_bin, n_bin=n_bin,
          m_synth=m_synth, m_weight=m_weight, std_weight=std_weight,
          transparent=False)




