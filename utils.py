import random
import itertools

import numpy as np
import scipy
from scipy.special import wofz
import pandas as pd

import plotly.express as px

import collections
from functools import partial 


def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def Poly(x, a, b):
    return (10**a)*(x - b)**2

# ------- Peaks -----------

def generate_example_raman(x, n=9,
                           scale_range=(4,5.4), 
                           shift_range=(-100, 190),
                           alpha_range=(.2,4),
                           gamma_range=(.2,4)):
    """Generates a synthetic raman signal.
    
    Generates a mixture of voigt profiles generated by combining gaussian and lorentzian peaks.

    Parameters
    ----------
    x
        Input array. Usually np.linspace array.
    n
        Number of peaks
    scale_range
        Intensity multiplier of the form `10**scale_range` will be randomly sampled from this range.
    shift_range
        Wavenumber shift will be randomly sampled from this range
    alpha_range
        Specifies the FWHM of the gaussian contribution in the voigt profile
    gamma_range
        Specifies the FWHM of the lorentzian contribution in the voigt profile

    Returns
    -------
    np.array
        Peak values over x.
        
    """


    scale = [random.uniform(*scale_range) for i in range(n)]
    shift = [random.uniform(*shift_range) for i in range(n)]

    alpha = [random.uniform(*alpha_range) for i in range(n)]
    gamma = [random.uniform(*gamma_range) for i in range(n)]

    funcs = [(10**scale_i)*V(x + shift_i, alpha_i, gamma_i) for scale_i, shift_i, alpha_i, gamma_i in zip(scale, shift, alpha, gamma)]

    f_sum = [sum(x) for x in zip(*funcs)]

    return np.array(f_sum)

# ------ Background ----------
def generate_example_background_gaussians(x, n=5,
                           scale_range=(6,7), 
                           shift_range=(-100, 190),
                           alpha_range=(100,300)):
    """Generates a synthetic background signal of `n` broad gaussian peaks.

    Parameters
    ----------
    x
        Input array. Usually np.linspace array.
    n
        Number of gaussian peaks
    scale_range
        Intensity multiplier of the form `10**scale_range` will be randomly sampled from this range.
    shift_range
        Wavenumber shift will be randomly sampled from this range.
    alpha_range
        Specifies the FWHM of the gaussians.

    Returns
    -------
    np.array
        Background values over x.
        
    """

    scale = [random.uniform(*scale_range) for i in range(n)]
    shift = [random.uniform(*shift_range) for i in range(n)]

    alpha = [random.uniform(*alpha_range) for i in range(n)]


    background_gaussians = [(10**scale_i)*G(x + shift_i, alpha_i) for scale_i, shift_i, alpha_i in zip(scale, shift, alpha)]
    combined_gaussians = [sum(i) for i in zip(*background_gaussians)]

    background_funcs = np.array(combined_gaussians)

    return background_funcs

def generate_example_background_polynomial(x, 
                           a_range=(-0.4,-0.3),
                           b_range=(150,300)):
    """Generates a synthetic background polynomial signal.

    The polynomial will be of the form
    $$$
    poly(x) = a*(x - b)**2
    $$$
    
    Parameters
    ----------
    x
        Input array. Usually np.linspace array.
    a_range
        Specifies the multiplier of the polynomial of the form 10**a_range.
    b_range
        Specifies the shift of the polynomial.

    Returns
    -------
    np.array
        Background values over x.
        
    """


    a = random.uniform(*a_range)
    b = random.uniform(*b_range)

    poly = Poly(x, a, b)

    background_funcs = np.array(poly)

    return background_funcs

def generate_example_background(x, *args, **kwargs):
    """Generates a synthetic background signal by combining `n` broad gaussian peaks with
    a polynomial.

    The polynomial will be of the form
    $$$
    poly(x) = a*(x - b)**2
    $$$
    
    Parameters
    ----------
    x
        Input array. Usually np.linspace array.
    n
        Number of gaussian peaks
    scale_range
        Intensity multiplier will be randomly sampled from this range.
    shift_range
        Wavenumber shift will be randomly sampled from this range.
    alpha_range
        Specifies the FWHM of the gaussians.
    a_range
        Specifies the multiplier of the polynomial.
    b_range
        Specifies the shift of the polynomial.

    Returns
    -------
    np.array
        Background values over x.
        
    """

    poly = generate_example_background_polynomial(x, *args, **kwargs)

    combined_gaussians = generate_example_background_gaussians(x, *args, **kwargs)

    background_funcs = np.array(poly) + np.array(combined_gaussians)

    return background_funcs

# ------ Noise -----------

def generate_example_noise(spectra, *args, **kwargs):
    """Generates poissonian noise over an input spectra.

    Parameters
    ----------
    spectra: np.array
        Input array. 
   
    Returns
    -------
    np.array
        Noisy version of input spectra.
        
    """

    noise_mask = np.random.poisson(spectra, *args, **kwargs)

    noisy_spectra = spectra + noise_mask

    return noisy_spectra
    
    return np.array(noisy_raman)

def generate_raman_example(x, *args, **kwargs):
    raman = generate_example_raman(x, *args, **kwargs) 

    background = generate_example_background(x, *args, **kwargs)

    noisy_raman = generate_example_noise(raman + background, *args, **kwargs)

    return noisy_raman

def generate_single_raman_example(x, 
scale, 
shift, 
alpha, 
gamma, 
a, 
b,
c,
scale_background,
alpha_background, 
shift_background,
*args, **kwargs):

    # raman = (10**scale)*V(x + shift, alpha, gamma)

    # poly = Poly(x, a, b, c)
    
    # background_gaussian = G(x + shift_background, alpha_background)

    # background_funcs = np.array(poly)*scale_background + np.array(background_gaussian)

    # background_funcs = background_funcs

    # noisy_raman = background_funcs#raman + background_funcs
    raman_signal = generate_example_raman(x) 
    background_signal = generate_example_background(x)
    combined_signal = raman_signal + background_signal
    noisy_combined_signal = generate_example_noise(combined_signal)
    return noisy_combined_signal


def generate_training_set(x, num_base_examples=1):
    
    """Will generate num_base_examples**2 total examples."""

    raman_examples = [generate_example_raman(x) for _ in range(num_base_examples)]

    background_examples = [generate_example_background(x) for _ in range(num_base_examples)]

    product_pairs = itertools.product(raman_examples,background_examples)

    training_set = [(generate_example_noise(raman + background), raman) for (raman, background) in product_pairs] 
    
    input, target = zip(*training_set)

    return input, target
