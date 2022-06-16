import numpy as np
import constants as ct


def qsat(tk):
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    return 640380.0 / np.exp(5107.4 / tk)


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
