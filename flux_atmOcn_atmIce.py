import os
import netCDF4
import numpy as np
import constants as ct
from utilities import *
import matplotlib.pyplot as plt


def flux_atmIce(mask, rbot, zbot, ubot, vbot, qbot, tbot, thbot, ts):
    """atm/ice fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (Pa)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N)
        tauy (:obj:`ndarray`): surface stress, maridional (N)
        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    vmag = np.maximum(ct.UMIN_I, np.sqrt((ubot[...])**2 + (vbot[...])**2))

    # virtual potential temperature (K)
    thvbot = thbot[...] * (1.0 + ct.ZVIR * qbot[...])

    # sea surface humidity (kg/kg)
    ssq = qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = np.log(zbot[...] / ct.ZREF)
    cp = ct.CPDAIR * (1.0 + ct.CPVIR * ssq[...])
    ct.LTHEAT = ct.LATICE + ct.LATVAP

    # First estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    rdn = ct.KARMAN / np.log(ct.ZREF / ct.ZZSICE)
    rhn = rdn
    ren = rdn

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thvbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift all coeffs to measurement height and stability
    rd = rdn / (1.0 + rdn / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn / (1.0 + rhn / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # Iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thvbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable[...] = 0.5 + 0.5 * np.sign(hol[...])
    xsq[...] = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq[...] = np.sqrt(xsq[...])
    psimh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift all coeffs to measurement height and stability
    rd[...] = rdn / (1.0 + rdn / ct.KARMAN * (alz[...] - psimh[...]))
    rh[...] = rhn / (1.0 + rhn / ct.KARMAN * (alz[...] - psixh[...]))
    re[...] = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar[...] = rd[...] * vmag[...]
    tstar[...] = rh[...] * delt[...]
    qstar[...] = re[...] * delq[...]

    # Compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * ubot[...] / vmag[...] * mask[...]
    tauy = tau[...] * vbot[...] / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ct.LTHEAT * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -ct.STEBOL * ts[...]**4 * mask[...]

    # water flux
    evap = lat[...] / ct.LTHEAT * mask[...]

    # compute diagnostic: 2m reference height temperature

    # compute function of exchange coefficients. Assume that
    # cn = rdn*rdn, cm=rd*rd and ch=rh*rd, and therefore
    # 1/sqrt(cn(n))=1/rdn and sqrt(cm(n))/ch(n)=1/rh
    bn = ct.KARMAN / rdn
    bh = ct.KARMAN / rh[...]

    # interpolation factor for stable and unstable cases
    ln0 = np.log(1.0 + (ct.ZTREF / zbot[...]) * (np.exp(bn) - 1.0))
    ln3 = np.log(1.0 + (ct.ZTREF / zbot[...]) * (np.exp(bn - bh[...]) - 1.0))
    fac = (ln0[...] - ct.ZTREF/zbot[...] * (bn - bh[...])) / bh[...] * stable[...]\
        + (ln0[...] - ln3[...]) / bh[...] * (1.0 - stable[...])
    fac = np.minimum(np.maximum(fac, 0.0), 1.0)

    # actual interpolation
    tref = (ts[...] + (tbot[...] - ts[...]) * fac[...]) * mask[...]
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, ustar, tstar, qstar)


def flux_atmOcn(mask, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts):
    """atm/ocn fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (kg/m^3)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N)
        tauy (:obj:`ndarray`): surface stress, maridional (N)

        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)
        duu10n(:obj:`ndarray`): diag: 10m wind speed squared (m/s)^2

        ustar_sv(:obj:`ndarray`): diag: ustar
        re_sv   (:obj:`ndarray`): diag: sqrt of exchange coefficient (water)
        ssq_sv  (:obj:`ndarray`): diag: sea surface humidity  (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    al2 = np.log(ct.ZREF / ct.ZTREF)

    vmag = np.maximum(ct.UMIN_O, np.sqrt((ubot[...] - us[...])**2
                                         + (vbot[...] - vs[...])**2))

    # sea surface humidity (kg/kg)
    ssq = 0.98 * qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff. (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = np.log(zbot[...] / ct.ZREF)
    cp = ct.CPDAIR * (1.0 + ct.CPVIR * ssq[...])

    # first estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    stable = 0.5 + 0.5 * np.sign(delt[...])
    rdn = np.sqrt(cdn(vmag[...]))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = 0.0346

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = np.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol[...] = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable[...] = 0.5 + 0.5 * np.sign(hol[...])
    xsq[...] = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq[...] = np.sqrt(xsq[...])
    psimh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh[...] = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd[...] = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn[...] = np.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn[...] = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd[...] = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh[...] = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re[...] = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar[...] = rd[...] * vmag[...]
    tstar[...] = rh[...] * delt[...]
    qstar[...] = re[...] * delq[...]

    # compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * (ubot[...] - us[...]) / vmag[...] * mask[...]
    tauy = tau[...] * (vbot[...] - vs[...]) / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ct.LATVAP * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -ct.STEBOL * ts[...]**4 * mask[...]

    # water flux
    evap = lat[...] / ct.LATVAP * mask[...]

    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared

    hol[...] = hol[...] * ct.ZTREF / zbot[...]
    xsq = np.maximum(1.0, np.sqrt(np.abs(1.0 - 16.0 * hol[...])))
    xqq = np.sqrt(xsq)
    psix2 = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])
    fac = (rh[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...])
    tref = thbot[...] - delt[...] * fac[...]

    # pot. temp to temp correction
    tref[...] = (tref[...] - 0.01 * ct.ZTREF) * mask[...]
    fac[...] = (re[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...]) * mask[...]
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    # 10m wind speed squared
    duu10n = u10n[...] * u10n[...] * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, duu10n, ustar, tstar, qstar)


def flux_atmOcnIce(mask, ps, qbot, rbot, ubot, vbot, tbot, us, vs, ts):
    """Calculates bulk net heat flux

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        ps   (:obj:`ndarray`): surface pressure (Pa)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        rbot (:obj:`ndarray`): atm density at full model level (kg/m^3)
        tbot (:obj:`ndarray`): temperature at full model level (K)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): surface temperature   (K)

    Returns:
        tuple(:obj:`ndarray`, :obj:`ndarray`, :obj:`ndarray`) 

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """

    vmag = np.maximum(ct.UMIN_O, np.sqrt((ubot[...] - us[...])**2
                                         + (vbot[...] - vs[...])**2))

    # long-wave radiation (IR)
    qir = -ct.STEBOL * ts[...]**4 * mask[...]

    # sensible heat flux
    qh = rbot[...] * ct.CPDAIR * ct.CH * vmag[...] * (tbot[...] - ts[...]) * mask[...]

    # latent heat flux
    qe = -rbot[...] * ct.CE * ct.LATVAP * vmag[...] * (qsat_august_eqn(ps, ts)
                                                       - qbot[...]) * mask[...]

    return (qir, qh, qe)


def main(itime):

    def read_forcing(var, file):
        with netCDF4.Dataset(file) as infile:
            return np.squeeze(infile[var][:].T)

    def plot(ds, fname, cmap=None, vmin=None, vmax=None):
        plt.figure(figsize=(10, 5))
        cs = plt.pcolor(longitude, latitude,
                        ds.T, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        shading='auto')
        plt.colorbar(cs)
        plt.savefig(f'{output_path}/{fname}.png')

    # Fields in the input file are defined on IFS model levels:
    #   levels 0 (L1)   - surfaces (lnsp & z)
    #   levels 1 (L136) - bottom - 1
    #   levels 2 (L137) - bottom
    #
    # Sigma coefficients hyam(i) & hybm(i) are given on
    #   L1 (TOA) - L137(8) (near-surface) model levels
    input_era5_ml = './era5/era5_198x_ml_4x4deg/era5_198x_ml_4x4deg_monthly_mean.nc'
    longitude = read_forcing('longitude', input_era5_ml)
    latitude = read_forcing('latitude', input_era5_ml)
    hyai = read_forcing('hyai', input_era5_ml)[-3:]
    hybi = read_forcing('hybi', input_era5_ml)[-3:]
    hyam = read_forcing('hyam', input_era5_ml)[-2:]   # L136-L137
    hybm = read_forcing('hybm', input_era5_ml)[-2:]   # L136-L137

    lnsp = read_forcing('lnsp', input_era5_ml)[..., 0, itime]
    ubot = read_forcing('u', input_era5_ml)[..., 1, itime]   # L136
    vbot = read_forcing('v', input_era5_ml)[..., 1, itime]   # L136
    q = read_forcing('q', input_era5_ml)[..., 1:, itime]     # L136-L137
    t = read_forcing('t', input_era5_ml)[..., 1:, itime]     # L136-L137
    qbot = q[..., 0]   # L136
    tbot = t[..., 0]   # L136

    input_era5_sfc = './era5/era5_198x_sfc_4x4deg/era5_198x_sfc_4x4deg_monthly_mean.nc'
    lsm = read_forcing('lsm', input_era5_sfc)[..., itime]
    siconc = read_forcing('siconc', input_era5_sfc)[..., itime]
    sst = read_forcing('sst', input_era5_sfc)[..., itime]
    tcc = read_forcing('tcc', input_era5_sfc)[..., itime]
    swr_net = read_forcing('msnswrf', input_era5_sfc)[..., itime]
    lwr_net = read_forcing('msnlwrf', input_era5_sfc)[..., itime]
    lwr_dw = read_forcing('msdwlwrf', input_era5_sfc)[..., itime]
    sshf_era5 = read_forcing('msshf', input_era5_sfc)[..., itime]
    slhf_era5 = read_forcing('mslhf', input_era5_sfc)[..., itime]

    input_oras5 = './oras5/'
    # (degC) --> (K)
    ts = read_forcing('votemper',
                      f'{input_oras5}'
                      f'votemper_control_monthly_highres_3D_19800{itime+1}_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0] + 273.15
    us = read_forcing('vozocrtx',
                      f'{input_oras5}'
                      f'vozocrtx_control_monthly_highres_3D_19800{itime+1}_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0]
    vs = read_forcing('vomecrty',
                      f'{input_oras5}'
                      f'vomecrty_control_monthly_highres_3D_19800{itime+1}_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0]

    sp = np.exp(lnsp)
    ph = get_press_levs(sp, hyai, hybi)
    pf = get_press_levs(sp, hyam, hybm)
    zbot = compute_z_level(t, q, ph)   # L136

    # air density
    rbot = ct.MWDAIR / ct.RGAS * pf[:, :, 0] / tbot[...]   # L136

    # potential temperature
    thbot = (tbot[...] * (ct.P0 / pf[:, :, 0])**ct.CAPPA)    # L136

    mask_nan = np.isnan(ts)
    mask_ice = np.zeros(siconc.shape)
    mask_ocn = np.zeros(lsm.shape)

    ts[mask_nan] = 0
    us[mask_nan] = 0
    vs[mask_nan] = 0

    mask_ice[siconc > 0.] = 1
    mask_ocn[lsm == 0.] = 1
    mask_ocn_ice = mask_ocn.copy()
    mask_ocn_ice[siconc > 0.] = 0

    atmOcn_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy',
                  'tref', 'qref', 'duu10n', 'ustar', 'tstar', 'qstar'),
             flux_atmOcn(mask_ocn_ice, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts)))

    atmIce_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy',
                  'tref', 'qref', 'ustar', 'tstar', 'qstar'),
             flux_atmIce(mask_ice, rbot, zbot, ubot, vbot, qbot, tbot, thbot, ts)))

    # Net LW radiation flux from sea surface
    lwnet_ocn = net_lw_ocn(mask_ocn_ice, latitude, qbot, sst, tbot, tcc)

    # Downward LW radiation flux over sea-ice
    lwdw_ice = dw_lw_ice(mask_ice, tbot, tcc)

    # Net surface radiation flux (without short-wave)
    qnet = swr_net + lwnet_ocn\
         + lwdw_ice + atmIce_fluxes['lwup']\
         + atmIce_fluxes['sen'] + atmOcn_fluxes['sen']\
         + atmIce_fluxes['lat'] + atmOcn_fluxes['lat']

    qir, qh, qe = flux_atmOcnIce(mask_ocn, sp, qbot, rbot, ubot, vbot, tbot, us, vs, ts)
    qnet_simple = swr_net + qir + lwr_dw + qh + qe 

    dqir_dt, dqh_dt, dqe_dt = dqnetdt(mask_ocn, sp, rbot, sst, ubot, vbot, us, vs)

    # ----------------------------------------------------------------

    output_path = './output' 
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    mask_ocn[lsm != 0.] = np.nan

    plot(rbot * mask_ocn, 'rbot', cmap='viridis', vmin=1., vmax=1.5)
    plot((thbot - 273.15) * mask_ocn, 'thbot', cmap='RdBu_r', vmin=-40, vmax=40)

    plot(tbot - 273.15, 'tbot', cmap='RdBu_r', vmin=-50, vmax=50)
    plot(np.where(ts - 273.15 < -1.8, -999, ts - 273.15),
                  'sst_m18', cmap='RdBu_r', vmin=-30, vmax=30)
    plot(ts - 273.15, 'sst', cmap='RdBu_r', vmin=-30, vmax=30)
    plot(zbot * mask_ocn, 'zbot_era5', cmap='viridis', vmin=15, vmax=35)

    plot(ubot * mask_ocn, 'ubot', cmap='RdBu_r', vmin=-10, vmax=10)
    plot(vbot * mask_ocn, 'vbot', cmap='RdBu_r', vmin=-10, vmax=10)
    plot(us * mask_ocn, 'ssu', cmap='RdBu_r', vmin=-1, vmax=1)
    plot(vs * mask_ocn, 'ssv', cmap='RdBu_r', vmin=-1, vmax=1)

    plot(qbot * mask_ocn, 'qbot', cmap='viridis')
    plot(mask_ocn_ice, 'mask_ocn_ice', cmap='viridis')

    plot(atmOcn_fluxes['ustar'] * mask_ocn, 'ustar_ocn', cmap='viridis', vmin=0, vmax=0.5)
    plot(atmOcn_fluxes['tstar'] * mask_ocn, 'tstar_ocn', cmap='viridis')
    plot(atmOcn_fluxes['qstar'] * mask_ocn, 'qstar_ocn', cmap='viridis')

    plot(lwr_net * mask_ocn, 'lwr_net_era5', cmap='RdBu_r', vmin=-100, vmax=100)
    plot(swr_net * mask_ocn, 'swr_net_era5', cmap='viridis', vmin=0, vmax=300)
    plot((swr_net + lwr_net) * mask_ocn, 'swr_lwr_net_era5',
         cmap='RdBu_r', vmin=-200, vmax=200)

    plot((sshf_era5 - atmIce_fluxes['sen'] - atmOcn_fluxes['sen']) * mask_ocn,
         'sshf_diff', cmap='RdBu_r', vmin=-50, vmax=50)
    plot((slhf_era5 - atmIce_fluxes['lat'] - atmOcn_fluxes['lat']) * mask_ocn, 
         'slhf_diff', cmap='RdBu_r', vmin=-100, vmax=100)

    plot(slhf_era5 * mask_ocn, 'slhf_ocn_era5', cmap='viridis', vmin=-200, vmax=0)
    plot(atmIce_fluxes['lat'] + atmOcn_fluxes['lat'] , 'slhf_ocn', cmap='viridis', vmin=-200, vmax=0)

    plot(sshf_era5 * mask_ocn, 'sshf_ocn_era5', cmap='viridis', vmin=-200, vmax=0)
    plot(atmIce_fluxes['sen'] + atmOcn_fluxes['sen'], 'sen_ocn', cmap='viridis', vmin=-200, vmax=0)

    plot(lwnet_ocn, 'lwnet_ocn', cmap='RdBu_r', vmin=-100, vmax=100)
    plot(atmIce_fluxes['lwup'] + lwdw_ice, 'lwnet_ice', cmap='RdBu_r', vmin=-100, vmax=100)

    plot(qnet  * mask_ocn, 'qnet', cmap='RdBu_r', vmin=-200, vmax=200)
    plot(-(dqir_dt + dqh_dt + dqe_dt) * mask_ocn, 'dqnet_dt', vmin=0, vmax=70)

    plot(qnet_simple * mask_ocn, 'qnet_simple', cmap='RdBu_r', vmin=-200, vmax=200)
    plot(qir * mask_ocn, 'lwup_simple', cmap='viridis')
    plot(qh * mask_ocn, 'sen_simple', cmap='viridis', vmin=-200, vmax=0)
    plot(qe * mask_ocn, 'slhf_simple', cmap='viridis', vmin=-200, vmax=0)

    print(f"Global hflux mean: {np.nanmean(qnet  * mask_ocn)}")
    #for fld in atmIce_fluxes:
    #    plot(atmIce_fluxes[fld] + atmOcn_fluxes[fld], fld)


if __name__ == "__main__":
    main(0)

