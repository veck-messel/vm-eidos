from math import sin, cos, sqrt, log, exp

import numpy as np

def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))

fwhm_constant = 2.0 * sqrt(2.0 * log(2))

def normalized_gaussian_pulse(x, fwhm, center=0.0):
    sigma = fwhm / fwhm_constant
    return np.exp(-(((x - center) ** 2.0) / (2.0 * (sigma ** 2.0))))

def normalized_gaussian_derivative_pulse(x, fwhm, center=0.0):
    sigma = fwhm / fwhm_constant
    return (
        exp((1.0 / 2.0) - ((x - center) ** 2.0) / (2.0 * sigma ** 2.0)) * (x - center)
    ) / sigma