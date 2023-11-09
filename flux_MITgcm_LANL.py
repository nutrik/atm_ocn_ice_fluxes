import os
import netCDF4
import numpy as npx
import constants as ct
import matplotlib.pyplot as plt

def bulkf_formula_lanl(uw, vw, ta, qa, tsf, iceornot):
    """Calculate bulk formula fluxes over open ocean or seaice

        wind stress = (ust,vst) = rhoA * Cd * Ws * (del.u,del.v)
        Sensib Heat flux = fsha = rhoA * Ch * Ws * del.T * CpAir
        Latent Heat flux = flha = rhoA * Ce * Ws * del.Q * Lvap
                        = -Evap * Lvap
        with Ws = wind speed = sqrt(del.u^2 +del.v^2) ;
            del.T = Tair - Tsurf ; del.Q = Qair - Qsurf
            Cd,Ch,Ce = transfer coefficient for momentum, sensible
                     & latent heat flux [no units]

    Arguments:
        uw (:obj:`ndarray`): zonal wind speed (at grid center) [m/s]
        vw (:obj:`ndarray`): meridional wind speed (at grid center) [m/s]
        ta (:obj:`ndarray`): air temperature   [K]     at height ht
        qa (:obj:`ndarray`): specific humidity [kg/kg] at heigth ht
        tsf(:obj:`ndarray`): sea-ice or sea surface temperature [K]
        iceornot (:obj:`ndarray`): 0=land, 1=open water, 1=sea-ice, 2=sea-ice with snow

    Returns:
        flwupa (:obj:`ndarray`): upward long wave radiation (>0 upward) [W/m2]
        flha   (:obj:`ndarray`): latent heat flux         (>0 downward) [W/m2]
        fsha   (:obj:`ndarray`): sensible heat flux       (>0 downward) [W/m2]
        df0dT  (:obj:`ndarray`): derivative of heat flux with respect to Tsf [W/m2/K]
        ust    (:obj:`ndarray`): zonal wind stress (at grid center)     [N/m2]
        vst    (:obj:`ndarray`): meridional wind stress (at grid center)[N/m2]
        evp    (:obj:`ndarray`): evaporation rate (over open water) [kg/m2/s]
        ssq    (:obj:`ndarray`): surface specific humidity          [kg/kg]
        dEvdT  (:obj:`ndarray`): derivative of evap. with respect to tsf [kg/m2/s/K]
    """

    # Compute turbulent surface fluxes
    ht =  2.
    zref = 10.
    zice = 0.0005
    aln = npx.log(ht / zref)
    czol = zref * ct.KARMAN * ct.G

    lath = npx.where(iceornot > 1., ct.LATVAP + ct.LATICE,
                     ct.LATVAP * npx.ones(ta.shape))

    # wind speed
    us = npx.sqrt(uw[...] * uw[...] + vw[...] * vw[...])
    usm = npx.maximum(us[...], 1.0)

    t0 = ta[...] * (1.0 + ct.ZVIR * qa[...])
    ssq = 3.797915 * npx.exp(lath[...] * (7.93252e-6 - 2.166847e-3 / tsf[...])) / 1013.

    deltap = ta[...] - tsf[...] + ct.GAMMA_BLK * ht
    delq = qa[...] - ssq[...]

    # initialize estimate exchange coefficients
    rdn = ct.KARMAN / npx.log(zref / zice)
    rhn = rdn
    ren = rdn
    # calculate turbulent scales
    ustar = rdn * usm[...]
    tstar = rhn * deltap[...]
    qstar = ren * delq[...]

    # iteration with psi-functions to find transfer coefficients
    for _ in range(5):
        huol = czol / ustar[...]**2 * (tstar[...] / t0 + qstar[...]/(1. / ct.ZVIR + qa[...]))
        huol = npx.minimum(npx.abs(huol[...]), 10.0) * npx.sign(huol[...])
        stable = 0.5 + 0.5 * npx.sign(huol[...])
        xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * huol[...])), 1.0)
        x = npx.sqrt(xsq[...])
        psimh = -5. * huol[...] * stable[...] + (1. - stable[...])\
              * (2. * npx.log(0.5 * (1. + x[...]))
                + 2. * npx.log(0.5 * (1. + xsq[...]))
                - 2. * npx.arctan(x[...]) + npx.pi * 0.5)
        psixh = -5. * huol[...] * stable[...] + (1. - stable[...])\
              *  (2. * npx.log(0.5 * (1. + xsq[...])))

        # update the transfer coefficients
        rd = rdn / (1. + rdn * (aln[...] - psimh[...]) / ct.KARMAN)
        rh = rhn / (1. + rhn * (aln[...] - psixh[...]) / ct.KARMAN)
        re = rh

        # update ustar, tstar, qstar using updated, shifted coefficients.
        ustar = rd[...] * usm[...]
        qstar = re[...] * delq[...]
        tstar = rh[...] * deltap[...]

    #tau = ct.RHOA * ustar[...]**2
    #tau = tau * us[...] / usm[...]
    csha = ct.RHOA * ct.CPDAIR * us[...] * rh[...] * rd[...]
    clha = ct.RHOA * lath[...] * us[...] * re[...] * rd[...]

    fsha = csha[...] * deltap[...]
    flha = clha[...] * delq[...]
    evp = -flha[...] / lath[...]

    # upward long wave radiation
    flwupa = npx.where(iceornot > 1., ct.ICE_EMISSIVITY * ct.STEBOL * tsf[...]**4,
                                      ct.OCEAN_EMISSIVITY * ct.STEBOL * tsf[...]**4)
    dflwupdt = npx.where(iceornot > 1., 4. * ct.ICE_EMISSIVITY * ct.STEBOL * tsf[...]**3,
                                        4. * ct.OCEAN_EMISSIVITY * ct.STEBOL * tsf[...]**3,)

    devdt = clha[...] * ssq[...] * 2.166847e-3 / (tsf[...] * tsf[...])
    dflhdt = -lath[...] * devdt[...]
    dfshdt = -csha[...]

    # total derivative with respect to surface temperature
    df0dt = -dflwupdt[...] + dfshdt[...] + dflhdt[...]

    #  wind stress at center points
    bulkf_cdn = 2.7e-3 / usm[...] + 0.142e-3 + 0.0764e-3 * usm[...]
    ust = ct.RHOA * bulkf_cdn * us[...] * uw[...]
    vst = ct.RHOA * bulkf_cdn * us[...] * vw[...]

    return (flwupa, flha, fsha, df0dt, ust, vst, evp, ssq, devdt)


def main(itime):

    def read_forcing(var, file):
        with netCDF4.Dataset(file) as infile:
            return npx.squeeze(infile[var][:].T)

    def plot(ds, fname, cmap=None, vmin=None, vmax=None):
        plt.figure(figsize=(10, 5))
        cs = plt.pcolor(longitude, latitude,
                        ds.T, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        shading='auto')
        plt.colorbar(cs)
        plt.savefig(f'{output_path}/{fname}.png')

    output_path = './output_mitgcm' 
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input_era5_ml = './era5/era5_198x_ml_4x4deg/era5_198x_ml_4x4deg_monthly_mean.nc'
    longitude = read_forcing('longitude', input_era5_ml)
    latitude = read_forcing('latitude', input_era5_ml)
    q10m = read_forcing('q', input_era5_ml)[..., -1, itime]   # near-surface only (not available q10m)

    input_era5_sfc = './era5/era5_198x_sfc_4x4deg/era5_198x_sfc_4x4deg_monthly_mean.nc'
    u10m = read_forcing('u10', input_era5_sfc)[..., itime]
    v10m = read_forcing('v10', input_era5_sfc)[..., itime]
    t2m = read_forcing('t2m', input_era5_sfc)[..., itime]
    sst = read_forcing('sst', input_era5_sfc)[..., itime]
    lsm = read_forcing('lsm', input_era5_sfc)[..., itime]
    siconc = read_forcing('siconc', input_era5_sfc)[..., itime]
    swr_net = read_forcing('msnswrf', input_era5_sfc)[..., itime]
    lwr_dw = read_forcing('msdwlwrf', input_era5_sfc)[..., itime]

    mask_ocn = npx.ones(lsm.shape)
    mask_ocn[siconc > 0.] = 2
    # mask_ocn[snowcover > 0.] = 3
    mask_ocn[lsm > 0.] = 0

    lwup, lat, sen, dqnetdt, taux, tauy, evap, ssq, devapdt\
       = bulkf_formula_lanl(u10m, v10m, t2m, q10m, sst, mask_ocn)

    swr_net = npx.where(mask_ocn == 1,
                        swr_net * (1. - ct.OCEAN_ALBEDO),
                        swr_net * (1. - ct.ICE_ALBEDO))
    qnet = swr_net - lwup + lwr_dw + sen + lat

    mask4plot = npx.ones(lsm.shape)
    mask4plot[mask_ocn == 0] = 0
    plot(mask4plot, 'mask4plot')
    plot(u10m * mask4plot, 'u10m', cmap='RdBu_r', vmin=-10, vmax=10)
    plot(v10m * mask4plot, 'v10m', cmap='RdBu_r', vmin=-10, vmax=10)
    plot((t2m - ct.TF0KEL) * mask4plot, 't2m', cmap='RdBu_r', vmin=-30, vmax=30)
    plot(q10m * mask4plot, 'q10m', cmap='viridis')
    plot((sst - ct.TF0KEL) * mask4plot, 'sst', cmap='RdBu_r', vmin=-30, vmax=30)
    plot((-lwup * mask4plot + lwr_dw * mask4plot), 'lw_net',
         cmap='RdBu_r', vmin=-100, vmax=100)
    plot(lat * mask4plot, 'lat', cmap='viridis', vmin=-300, vmax=0)
    plot(sen * mask4plot, 'sen', cmap='viridis', vmin=-200, vmax=0)
    plot(-dqnetdt * mask4plot, 'dqnetdt', cmap='viridis', vmin=0, vmax=150)
    plot(qnet * mask4plot, 'qnet', cmap='RdBu_r', vmin=-200, vmax=200)
    plot(taux * mask4plot, 'ust', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plot(tauy * mask4plot, 'vst', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plot(evap * mask4plot, 'evap', cmap='viridis')
    plot(ssq * mask4plot, 'ssq', cmap='viridis')
    plot(devapdt * mask4plot, 'devapdt', cmap='viridis')


if __name__ == "__main__":
    a = npx.array([500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 
    11500, 12500, 13500, 14500, 15500, 16509.84, 17547.9, 18629.13, 19766.03, 
    20971.14, 22257.83, 23640.88, 25137.02, 26765.42, 28548.37, 30511.92, 
    32686.8, 35109.35, 37822.76, 40878.46, 44337.77, 48273.67, 52772.8, 
    57937.29, 63886.26, 70756.33, 78700.25, 87882.52, 98470.59, 110620.4])
    print(len(a))
    print(a[40])
    main(0)
