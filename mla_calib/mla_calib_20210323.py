#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLA amplitude and phase lag calibration values.
Measured on 23.03.2021. check the labnotebook for details



"""

import numpy as np

# ===========================================================================
# Measured amplitude and phase values 
# ===========================================================================
amplitude_meas = [
    38.5,
    77.2,
    117.0,
    156,
    196,
    236,
    279,
    320,
    361,
    402,
    443,
    485,
    526,
    566,
    607,
    648,
    689,
    730,
    770,
    810,
    851,
    890,
    932,
    971,
    1010,
    1050,
    1090,
    1130,
    1170,
    1210,
    1240
]

phase_meas = [
    89.7,
    89.7,
    89.4,
    89.0,
    88.2,
    87.5,
    87.0,
    86.2,
    85.4,
    84.6,
    83.9,
    83.0,
    82.3,
    81.5,
    80.0,
    79.9,
    79.2,
    78.4,
    77.7,
    76.9,
    76.1,
    75.4,
    74.7,
    73.9,
    73.2,
    72.4,
    71.7,
    70.9,
    70.2,
    69.5,
    68.8
]


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
DEVIATION = 0

# AMPLITUDE_LAG = calculate_amplitude_lag(amplitude_meas)
# PHASE_LAG = calculate_phase_lag(phase_meas, deviation=DEVIATION)

# for val in AMPLITUDE_LAG:
#     print('\t{:.6f},'.format(val))
    
# for val in PHASE_LAG:
#     print('\t{:.1f},'.format(val))

# ===========================================================================
# Hardcode the calculated values 
# ===========================================================================


AMPLITUDE_LAG = np.array([
    # 1.000000,
    # 0.997409,
    # 0.987179,
    # 0.987179,
    # 0.982143,
    # 0.978814,
    # 0.965950,
    # 0.962500,
    # 0.959834,
    # 0.957711,
    # 0.955982,
    # 0.952577,
    # 0.951521,
    # 0.952297,
    # 0.951400,
    # 0.950617,
    # 0.949927,
    # 0.949315,
    # 0.950000,
    # 0.950617,
    # 0.950059,
    # 0.951685,
    # 0.950107,
    # 0.951596,
    # 0.952970,
    # 0.953333,
    # 0.953670,
    # 0.953982,
    # 0.954274,
    # 0.954545,
    # 0.962500,
	1.000000,
	1.002597,
	1.012987,
	1.012987,
	1.018182,
	1.021645,
	1.035250,
	1.038961,
	1.041847,
	1.044156,
	1.046045,
	1.049784,
	1.050949,
	1.050093,
	1.051082,
	1.051948,
	1.052712,
	1.053391,
	1.052632,
	1.051948,
	1.052566,
	1.050767,
	1.052513,
	1.050866,
	1.049351,
	1.048951,
	1.048581,
	1.048237,
	1.047918,
	1.047619,
	1.038961,
])


PHASE_LAG = np.array([
    0.0,
    0.0,
    0.3,
    0.7,
    1.5,
    2.2,
    2.7,
    3.5,
    4.3,
    5.1,
    5.8,
    6.7,
    7.4,
    8.2,
    9.7,
    9.8,
    10.5,
    11.3,
    12.0,
    12.8,
    13.6,
    14.3,
    15.0,
    15.8,
    16.5,
    17.3,
    18.0,
    18.8,
    19.5,
    20.2,
    20.9,
])




