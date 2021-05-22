#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLA amplitude and phase lag calibration values.
Measured on 21.05.2021. check the labnotebook for details



"""

import os

import numpy as np


# ===========================================================================
# Loading function
# ===========================================================================

def load_calib_data(txt_fn):
    """loads and returns the frequency, amplitude, and phase values from the 
    given csv file
    
    """    
    arr = np.loadtxt(txt_fn, skiprows=1, delimiter=',')
    f, amp, phase = arr.T
    return f, amp, phase


# ===========================================================================
# Load measured amplitude and phase values 
# ===========================================================================
txt_fn = 'mla_calib_20210521.csv'

f, amplitude_meas, phase_meas = load_calib_data(txt_fn)


# ===========================================================================
# lag-calculation function
# ===========================================================================

def calculate_phase_lag(phase_meas, deviation=0):
    
    phase_lag = phase_meas[0] - np.array(phase_meas)
    phase_lag[0] = -1*deviation
    return phase_lag


def calculate_amplitude_lag(amplitude_meas):
    
    amp_base_tone = amplitude_meas[0]
    tone_nr = np.arange(len(amplitude_meas))+1
    amp_rescaled = np.array(amplitude_meas) / tone_nr
#    amplitude_lag = amp_base_tone / amp_rescaled
    amplitude_lag = amp_rescaled / amp_base_tone
    
    return amplitude_lag



# ===========================================================================
# Calculate the amplitude and phase lag values
# ===========================================================================
# DEVIATION = 0

# AMPLITUDE_LAG = calculate_amplitude_lag(amplitude_meas)
# PHASE_LAG = calculate_phase_lag(phase_meas, deviation=DEVIATION)

# print('AMPLITUDE LAG:')
# for val in AMPLITUDE_LAG:
#     print('\t{:.6f},'.format(val))

# print('')
# print('PHASE LAG:')
# for val in PHASE_LAG:
#     print('\t{:.1f},'.format(val))

# ===========================================================================
# Hardcode the calculated values 
# ===========================================================================


AMPLITUDE_LAG = np.array([
	1.000000,
	1.000000,
	1.001873,
	1.004213,
	1.011236,
	1.015918,
	1.023274,
	1.028792,
	1.036205,
	1.039326,
	1.044433,
	1.048689,
	1.047969,
	1.053371,
	1.054307,
	1.053371,
	1.049240,
	1.047129,
	1.043761,
	1.039326,
	1.033975,
	1.030388,
	1.018564,
	1.015918,
	1.005618,
	0.996111,
	0.988348,
	0.978130,
	0.967648,
	0.955056,
	0.942370,
])


PHASE_LAG = np.array([
	0.0,
	1.1,
	2.0,
	2.9,
	4.2,
	5.3,
	6.7,
	8.0,
	9.6,
	11.1,
	13.3,
	14.4,
	16.2,
	18.0,
	19.9,
	21.6,
	23.5,
	25.2,
	27.0,
	28.7,
	30.5,
	32.5,
	34.4,
	35.9,
	37.6,
	39.4,
	41.0,
	42.9,
	44.1,
	45.8,
	47.3,
])




