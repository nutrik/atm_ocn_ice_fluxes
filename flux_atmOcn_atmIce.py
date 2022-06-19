import numpy as np
import constants as ct
from utilities import *


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
        cesm v.1.0.5: models/csm_share/shr/shr_flux_mod.F90
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

    return (sen, lat, lwup, evap, taux, tauy, tref, qref)


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
        cesm v.1.0.5: models/csm_share/shr/shr_flux_mod.F90
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

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, duu10n)


def main(itime):
    import netCDF4
    import matplotlib.pyplot as plt

    def read_forcing(var, file):
        with netCDF4.Dataset(file) as infile:
            return np.squeeze(infile[var][:].T)

    # Fields in the input file are defined on IFS model levels:
    #   levels 0 (L1)   - surfaces (lnsp & z)
    #   levels 1 (L136) - bottom - 1
    #   levels 2 (L137) - bottom
    #
    # Sigma coefficients hyam(i) & hybm(i) are given on
    #   L1 (TOA) - L137(8) (near-surface) model levels
    input_era5_ml = './era5/era5_200x_ml_4x4deg/era5_200x_ml_4x4deg.nc'
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

    input_era5_sfc = './era5/era5_200x_sfc_4x4deg/era5_200x_sfc_4x4deg.nc'
    lsm = read_forcing('lsm', input_era5_sfc)[..., itime]
    siconc = read_forcing('siconc', input_era5_sfc)[..., itime]

    input_oras5 = './oras5/'
    # (degC) --> (K)
    ts = read_forcing('votemper',
                      f'{input_oras5}'
                      'votemper_control_monthly_highres_3D_200001_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0] + 273.15
    us = read_forcing('vozocrtx',
                      f'{input_oras5}'
                      'vozocrtx_control_monthly_highres_3D_200001_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0]
    vs = read_forcing('vomecrty',
                      f'{input_oras5}'
                      'vomecrty_control_monthly_highres_3D_200001_'
                      'CONS_v0.1_regrided_4x4deg.nc')[..., 0]

    sp = np.exp(lnsp)
    ph = get_press_levs(sp, hyai, hybi)
    pf = get_press_levs(sp, hyam, hybm)
    zbot = compute_z_level(t, q, ph)   # L136

    # air density
    rbot = (ct.RGAS / ct.MWDAIR * tbot[...] / pf[:, :, 0])   # L136

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
    mask_ocn[siconc > 0.] = 0

    atmOcn_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy', 'tref', 'qref', 'duu10n'),
             flux_atmOcn(mask_ocn, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts)))

    atmIce_fluxes =\
        dict(zip(('sen', 'lat', 'lwup', 'evap', 'taux', 'tauy', 'tref', 'qref'),
             flux_atmIce(mask_ice, rbot, zbot, ubot, vbot, qbot, tbot, thbot, ts)))

    for fld in atmIce_fluxes:
        plt.figure(figsize=(10, 5))
        cs = plt.pcolor(longitude, latitude,
                        (atmIce_fluxes[fld] + atmOcn_fluxes[fld]).T,
                        shading='auto')
        plt.colorbar(cs)
        plt.title(fld)
        plt.savefig(f'{fld}.png')


if __name__ == "__main__":
    main(0)
