"""
analytic_1D.py
jeudi 8 octobre 2020, 14:36:22 (UTC+0200)


"""
# ----------------------------------------------------------------------------
import numpy as np


# ----------------------------------------------------------------------------
def MT1D_analytic(thick, rho, per):
    """ Analytical forward modelling for MT1D """
    if len(thick) == len(rho):
        thick = thick[0 : -1]

    nlay = len(rho)
    frequencies = 1 / per
    amu = 4 * np.pi * 10**(-7) # Magnetic Permeability (H/m)
    Z = np.empty(len(per), dtype=complex)
    arho = np.empty(len(per))
    phase = np.empty(len(per))
    for iff, frq in enumerate(frequencies):
        nlay = len(rho)
        w =  2 * np.pi * frq
        imp = list(range(nlay))
        # compute basement impedance
        imp[nlay-1] = np.sqrt(w * amu * rho[nlay - 1] * 1j)
        for j in range(nlay-2, -1, -1):
            rholay = rho[j]
            thicklay = thick[j]
            # 3. Compute apparent rholay from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt((w * amu * (1 / rholay)) * 1j)
            wj = dj * rholay
            ej = np.exp(-2 * thicklay * dj)
            # The next step is to calculate the reflection coeficient (F6) 
            # and impedance (F7) using the current layer intrinsic impedance 
            # and the prior computer layer impedance j+1.
            belowImp = imp[j+1]
            rj = (wj - belowImp)/(wj + belowImp)
            re = rj*ej
            Zj = wj * ((1 - re)/(1 + re))
            imp[j] = Zj

        # Finally you can compute the apparent rholay F8 and phase F9 and print the resulting data!
        Z[iff] = imp[0]
        absZ = abs(Z[iff])
        arho[iff] = (absZ * absZ) / (amu * w)
        phase[iff] = np.arctan2(np.imag(Z[iff]), np.real(Z[iff])) * 180 / np.pi
        # if convert to microvolt/m/ntesla
        Z[iff] = Z[iff] / np.sqrt(amu * amu *10**6)

    return Z, arho, phase



