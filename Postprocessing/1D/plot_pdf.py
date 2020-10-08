"""
------------------------------------------------------------------------------
plot_pdf.py
jeudi 8 octobre 2020, 12:01:11 (UTC+0200)

script to plot marginal laws, parameter space exploration, synthetic model,
<m> and mbest

------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

# ----------------------------------------------------------------------------

def plot_pdfm(gen_name='pdf', pdf_m=None, x_bin=None, n_bin=None,
              m_synth=None, m_weight=None, transparent=True, **kwargs):
    """
    plot marginal laws, parameter space exploration, synthetic model,
    <m> and mbest
    
    arguments:
      gen_name: generic figure name. Path + /pdf_myrun...
      m_synth : synthetic model

    """
    nparam = m_synth.shape[0]
    for ipar in range(nparam):
        fig = plt.figure()
        valpar = round(m_synth[ipar], 2)
        meanpar = round(m_weight[ipar], 2)
        Sbin = sum(n_bin[ipar, :])
        hist = np.empty(int(Sbin))
        for j in range(len(n_bin[ipar, :])):
            hist[int(np.sum(n_bin[ipar, 0:j])) : int(np.sum(n_bin[ipar, 0:j+1]))] = x_bin[ipar, j]
        ax1 = fig.add_subplot(111)
        binplot = np.hstack((x_bin[ipar, :], x_bin[ipar, -1] \
                + np.diff(x_bin[ipar, :])[-1])) - np.diff(x_bin[ipar, :])[0]/2.
        ax1.hist(hist,bins=binplot)
        ax1.set_ylabel('Nbr of model', color='b', fontsize=10)
        ax1.set_xlabel('Resistivity (Log-scale)', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        ax2 = ax1.twinx()
        ax2.axvline(x=valpar, color='y', label='synth')
        ax2.axvline(x=meanpar, color='g', label='mean')
        ax2.plot(x_bin[ipar, :], pdf_m[ipar, :], 'r')
        ax2.set_ylabel('Probability', color='r',fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
        align_yaxis(ax1, 0, ax2, 0)
        ax2.legend(loc=0, fontsize=10)
        plt.suptitle('Parameter '+ str(ipar+1)+' <Rho>:' + str(meanpar) + '$\Omega.m$ (log scale), STD:' + str(round(std_weight[ipar], 2)), fontsize=12)
        plt.savefig(gen_name + '_ip_' + str(ipar+1) + '.png', transparent=transparent)
        plt.clf()

    return None

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
          m_synth=m_synth, m_weight=m_weight, transparent=False)




