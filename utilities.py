import numpy as np
import constants as ct


_cc = np.array([0.88, 0.84, 0.80,
                0.76, 0.72, 0.68,
                0.63, 0.59, 0.52,
                0.50, 0.50, 0.50,
                0.52, 0.59, 0.63,
                0.68, 0.72, 0.76,
                0.80, 0.84, 0.88])

_clat = np.array([-90.0, -80.0, -70.0,
                  -60.0, -50.0, -40.0,
                  -30.0, -20.0, -10.0,
                  -5.0,   0.0,   5.0,
                  10.0,  20.0,  30.0,
                  40.0,  50.0,  60.0,
                  70.0,  80.0,  90.0])


def qsat(tk):
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    return 640380.0 / np.exp(5107.4 / tk)


def qsat_august_eqn(ps, tk):
    """Saturated specific humidity (kg/kg)

    Arguments:
        ps (:obj:`ndarray`): atm sfc pressure (Pa)
        tk (:obj:`ndarray`): atm temperature (K)

    Returns:
        :obj:`ndarray`

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """
    return 0.622 / ps * 10**(9.4051 - 2353. / tk) * 133.322


def dqnetdt(mask, ps, rbot, sst, ubot, vbot, us, vs):
    """Calculates correction term of net ocean heat flux (W/m^2)

    Arguments:
        mask (:obj:`ndarray`): ocean mask (0-1)
        ps (:obj:`ndarray`): surface pressure (Pa)
        rbot (:obj:`ndarray`): atm density at full model level (kg/m^3)
        sst (:obj:`ndarray`): surface temperature (K)
        vmag (:obj:`ndarray`): atm wind speed at full model level (m/s)

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

    dqir_dt = -ct.STEBOL * 4. * sst[...]**3 * mask  # long-wave radiation correction (IR)
    dqh_dt = -rbot[...] * ct.CPDAIR * ct.CH * vmag[...] * mask  # sensible heat flux correction
    dqe_dt = -rbot[...] * ct.CE * ct.LATVAP * vmag[...] * 2353.\
        * np.log(10.) * qsat_august_eqn(ps, sst) / (sst[...]**2) * mask  # latent heat flux correction

    return (dqir_dt, dqh_dt, dqe_dt)


def net_lw_ocn(mask, lat, qbot, sst, tbot, tcc):
    """Compute net LW (upward - downward) radiation at the ocean surface (W/m^2)

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask        0 <=> out of domain
        lat (:obj:`ndarray`): latitude coordinates    (deg)
        qbot (:obj:`ndarray`): atm specific humidity  (kg/kg)
        sst (:obj:`ndarray`): sea surface temperature (K)
        tbot (:obj:`ndarray`): atm T                  (K)
        tcc (:obj:`ndarray`): total cloud cover       (0-1)

    Returns:
        :obj:`ndarray`

    Reference:
        Clark, N.E., L.Eber, R.M.Laurs, J.A.Renner, and J.F.T.Saur, (1974):
        Heat exchange between ocean and atmosphere in the eastern North Pacific for 1961-71,
        NOAA Technical report No. NMFS SSRF-682.
    """

    ccint = np.zeros(lat.shape)

    for i in range(20):
        idx = np.squeeze(np.argwhere((lat[:] > _clat[i]) & (lat[:] <= _clat[i+1])))
        ccint[idx] = _cc[i] + (_cc[i+1] - _cc[i])\
                   * (lat[idx] - _clat[i]) / (_clat[i+1] - _clat[i])

    frac_cloud_cover = 1. - ccint[np.newaxis, :] * tcc[...]**2
    rtea = np.sqrt(1000. * qbot[...] / (0.622 + 0.378 * qbot[...]) + ct.EPS2)

    return -ct.EMISSIVITY * ct.STEBOL * tbot[...]**3\
        * (tbot[...] * (0.39 - 0.05 * rtea[...]) * frac_cloud_cover
           + 4. * (sst[...] - tbot[...])) * mask[...]


def dw_lw_ice(mask, tbot, tcc):
    """Compute LW downward flux ove sea-ice (W/m^2)

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask        0 <=> out of domain
        tbot (:obj:`ndarray`): atm T                  (K)
        tcc (:obj:`ndarray`): total cloud cover       (0-1)

    Returns:
        :obj:`ndarray`

    Reference:
        Parkinson, C.L. and W.M. Washington, (1979): JGR,
        Vol.84, Issue C1, pp.311-337
        https://doi.org/10.1029/JC084iC01p00311
    """

    return ct.STEBOL * tbot[...]**4\
        * (1. - 0.261 * np.exp(-7.77e-4 * (273.15 - tbot[...])**2))\
        * (1. + 0.275 * tcc[...]) * mask[...]


def cdn(umps):
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    return 0.0027 / umps + 0.000142 + 0.0000764 * umps


def psimhu(xd):
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return np.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)\
        - 2.0 * np.arctan(xd) + 1.571


def psixhu(xd):
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return 2.0 * np.log((1.0 + xd * xd) / 2.0)


def get_press_levs(sp, hya, hyb):
    """Compute pressure levels

    Arguments:
        sp (:obj:`ndarray`): Atmospheric surface pressure
        hya (:obj:`ndarray`): Hybrid sigma level A coefficient for vertical grid
        hyb (:obj:`ndarray`): Hybrid sigma level B coefficient for vertical grid

    Return:
        :obj: `ndarray`
    """
    return hya[np.newaxis, np.newaxis, :]\
        + hyb[np.newaxis, np.newaxis, :] * sp[:, :, np.newaxis]


def compute_z_level(t, q, ph):
    """Computes the altitudes at ECMWF Integrated Forecasting System
    (ECMWF-IFS) model half- and full-levels (for 137 levels model reanalysis: L137)

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperture [K]
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg]
        ph (:obj:`ndarray`): Pressure at half model levels

    Note:
        The top level of the atmosphere is excluded

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/
          9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/CKB/
          ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    Returns:
        :obj:`ndarray`: Altitude of the atmospheric near surface layer (second IFS level)

    """

    # virtual temperature (K)
    tv = t[...] * (1.0 + ct.ZVIR * q[...])

    # compute geopotential for 2 lowermost (near-surface) model levels
    dlog_p = np.log(ph[:, :, 1:] / ph[:, :, :-1])
    alpha = 1. - ((ph[:, :, :-1] / (ph[:, :, 1:] - ph[:, :, :-1])) * dlog_p)
    tv = tv * ct.RDAIR

    # z_h is the geopotential of 'half-levels'
    # integrate z_h to next half level
    increment = np.flip(tv * dlog_p, axis=2)
    zh = np.cumsum(increment, axis=2)

    # z_f is the geopotential of this full level
    # integrate from previous (lower) half-level z_h to the
    # full level
    increment_zh = np.insert(zh, 0, 0, axis=2)
    zf = np.flip(tv * alpha, axis=2) + increment_zh[:, :, :-1]

    alt = ct.RE * zf / ct.G / (ct.RE - zf / ct.G)

    return alt[:, :, -1]
