m#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to generate mla simulation 

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d






# ===========================================================================
# simulation function
# ===========================================================================

# ===========================================================================
# auxiliary functions
# ===========================================================================
def interpolate_test_signal(x, y, **kwargs):
    """returns interpolation function of given x and y dataset"""
    interp_sttngs = {
        'kind':             'cubic',
        'assume_sorted':    False,
    }
    interp_sttngs.update(kwargs)
    return interp1d(x, y, **interp_sttngs)


def get_mla_modulation_prm(f_mod, amp, offset, n_harmonic, f_sample, n_period):
    """returns dictionary of mla modulation & sampling parameter """

    t_mod = 1./f_mod
    t_sample = 1./f_sample
    n_per_mod = int(t_mod/t_sample)
    nsamples = n_per_mod*n_period
    

    mod_prm = {
        'f_mod':        f_mod,
        't_mod':        t_mod,
        'amp':          amp,
        'offset':       offset,
        'n_harmonic':    n_harmonic,
        'f_sample':      f_sample,
        't_sample':      t_sample,
        'n_per_mod':     n_per_mod,
        'nsamples':      nsamples
    }
    return mod_prm

def get_time_vector(nsamples, t_sample):
    return np.arange(0, nsamples) * t_sample
    
    

def get_voltage_modulation(t_arr, amp, f_mod, offset):
    """ """
    def v_mod(t):
        return amp*np.cos(2*np.pi*f_mod*t) + offset
    
    return v_mod(t_arr)


def get_modulated_current(en, curr_, amp, f_mod, offset, nsamples, t_sample, **kwargs):
    
    interp_sttngs = {
        'kind':             'cubic',
        'assume_sorted':    False,
    }
    
    # get time vector    
    t_ = get_time_vector(nsamples, t_sample)
    
    # get voltage modulation
    v_t = get_voltage_modulation(t_, amp, f_mod, offset)
    
    # get current interpolation
    iv_trace = interpolate_test_signal(en, curr_, **interp_sttngs)
    
    # get current time trace
    i_t = iv_trace(v_t)
    
    return v_t, i_t

# ===========================================================================
# plot functions
# ===========================================================================

def plot_test_signal_interp(x, y, f, fg_st={}, ax_st={}, ln_st={}):
    """ """
        
    ln1_st = ln_st.copy()
    ln1_st.update({
         'label':        'test signal'
    })
    ln2_st = ln_st.copy()
    ln2_st.update({
        'label':         'spline interpolation'
    })
    
    x_ = np.linspace(x.min(), x.max(), 1001)
    
    fig = plt.figure(num=99); plt.clf()
    ax1 = fig.add_subplot(111, **ax_st)
    
    ln1 = ax1.plot(x, y, **ln1_st)
    ln2 = ax1.plot(x_, f(x_), **ln2_st)
    plt.legend(loc=0)
    plt.grid()
    return fig, ax1, [ln1, ln2]




