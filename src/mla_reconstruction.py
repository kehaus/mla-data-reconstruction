#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:15:39 2021

@author: kh
"""


__author__ = "kha"
__version__ = "1.0.0"


import os 
import sys
import cmath
import itertools

import numpy as np
from scipy import interpolate
import h5py


# ===========================================================================
# static MLA compensation values
# ===========================================================================
DEVIATION = 0
PHASE_LAG =  np.array([
    DEVIATION * -1, 
    1.1, 
    2.2, 
    3.4, 
    4.6, 
    6.2, 
    7.7, 
    9.4, 
    11.1, 
    13, 
    14.7, 
    16.6, 
    18.4, 
    20.2, 
    22, 
    23.8, 
    25.6, 
    27.4, 
    29.1, 
    30.8, 
    32.5, 
    34.1, 
    35.8, 
    37.5, 
    39, 
    40.6, 
    42.1, 
    43.7, 
    45.1, 
    46.6, 
    48
])
AMPLITUDE_LAG = np.array([
    1, 
    1.00142,
    0.99906, 
    0.99437, 
    0.98603, 
    0.98512, 
    0.97668, 
    0.97045, 
    0.9686, 
    0.96185, 
    0.96352, 
    0.96492, 
    0.96611, 
    0.96712, 
    0.97156, 
    0.97547, 
    0.98056, 
    0.98665, 
    0.99363, 
    1.00142, 
    1.00857, 
    1.01782, 
    1.02642, 
    1.0357, 
    1.04562, 
    1.05737, 
    1.0685, 
    1.0814, 
    1.0937, 
    1.10658, 
    1.12121
])



DATA_BLOCK_HEADER = "# of trigger:"
DATA_BLOCK_DELIMITER = ['\n', '']


# ===========================================================================
# load and parse functions
# ===========================================================================
class MLAReconstructionException(Exception):
    """ """
    pass

# ===========================================================================
# load and parse functions
# ===========================================================================

def _load_mla_data(mla_data_fn, mode='r'):
    """returns the loaded mla data as a list of data file lines
    
    Parameter
    ---------
        mla_data_fn | str
            filename of the MLA raw data file to be imported 
            (e.g. ``Measurement of 2021-03-13 0242.txt``)
        mode | char
            specifies the mode in which the file is opened. Check ``open()``
            for details
            
    Returns
    -------
        lines | list
            list where every element is a line from the provided data file
            
    Examples
    --------
        >>> fn = Measurement of 2021-03-13 0242.txt
        >>> lines = _load_mla_data(fn, mode='r')
            
    
    """
    f = open(mla_data_fn, mode)
    lines = f.readlines()
    f.close()
    return lines

def _create_block_list(lines):
    """returns the mla data as nested list where one list element represents 
    one block of data
    
    Function parses the provided list of lines, groups them into blocks and 
    returns them as a list of blocks. The first list element is the header 
    block. The following blocks are data blocks where one block corresponds
    to a recorded spectrum.
    
    Parameter
    ---------
        lines | list
            list where every element is a line from the provided data file. 
            This variable is produced from the ``_load_mla_data`` function.
            
    Returns
    -------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. list element) consists of a list of lines belonging
            to that block of data.
    
    
    """
    block_list = [[]]

    for line in lines:
        if line != '\n':
            block_list[-1].append(line)
        else:
            block_list.append([])
    return block_list

def _parse_mla_data_header(block_list, pixel_number=None):
    """returns a dictionary with the measurement parameters parsed from the 
    given header_block
    
    This funtions parses measurement parameter like ``modamp``, ``nsamples``,
    and ``demod_freqs`` from the header block in the provided block list.
    The header block is selected as the first element in the block list. 
    it is important that this is actually the header block otherwise this 
    function will raise an ``MLAReconstructionException``.
    
    Parameter
    ---------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
        pixel_number | int
            specifies the number of measurement blocks present in the mla data
            file. This parameter can be omitted if the specified ``block_list``
            contains all the measured data blocks.
            
    Returns
    -------
        prm | dict
            contains the measurement parameter present in the datafile header
            block. 
    
    Example
    -------
        Parse measurement parameter from the complete ``block_list`` of a 
        measurement raw data file
        
        >>> lines = _load_mla_data('mla_data.txt')
        >>> block_list = _create_block_list(lines)
        >>> hd_prm = _parse_mla_data_header(block_list)
    
        Parse parameter from an opened measurement raw data file. This method
        requires the knowledge of the ``pixel_number`` in advance
        
        >>> fn = 'Measurement of 2021-03-20 0655.txt'
        >>> pixel_number = 1000
        >>> f = open(fn, mode='r')
        >>> hd_block_lines = _read_one_block(f)
        >>> block_list = _create_block_list(hd_block_lines)
        >>> prm = _parse_mla_data_header(block_list, pixel_number=pixel_number)
    
    Raises
    ------
        MLAReconstructionException
            if given ``block_list`` does not contain a header block
    
    """
    header_block = block_list[0]
    prm = {}
    for line in header_block:
        if pixel_number != None:
            prm['pixelNumber'] = pixel_number
        else:
            data_block_list = _get_data_block_list(block_list)
            prm['pixelNumber'] = len(data_block_list)
        
        if "nsamples" in line:
            prm['nsamples'] = int(line[11:17])
        if "srate" in line:
            prm['srate'] = int(line[8:15])
        if "bandwidth" in line:
            prm['df'] = int(line[24:28])
        if "amplitude" in line:
            prm['modamp'] = float(line[23:26])
        if "number" in line:
            prm['demodnum'] = int(line[37:39])
        if "offset" in line:
            prm['offset'] = float(line[9:14])
            
    prm['demod_freqs'] = _parse_mla_demod_frequencies(header_block)
    
    if len(prm) == 0:
        raise MLAReconstructionException(
            """Given block_list does not have a header block. parsing failed"""
        )
    return prm

def _count_pixelNumber(mla_txt_fn, mode='r', verbose=False):
    """ """
    pixelNumber = -1 # start at -1 because there is n+1 '\n' for n data blocks
    
    f = open(mla_txt_fn, mode)
    s_ = 'start'
    while s_ != '':
        s_ = f.readline()
        if s_ == '\n':
            pixelNumber += 1
            if verbose: 
                if pixelNumber%1000 == 0:
                    print('pixelNumber: ', pixelNumber)

    f.close()
    
    return pixelNumber
    
# def _load_mla_data_header_(mla_txt_fn, pixel_number=None, mode='r', 
#                           readline_nr=100, verbose=False):
#     """ 
    
#     **OBSOLETE**
    
#     """
    
#     # load first lines from file
#     f = open(mla_txt_fn, mode)
#     lines = []
#     for i in range(readline_nr):
#         lines.append(f.readline())
#     f.close()

#     # create block lists:
#     block_list = _create_block_list(lines)
#     header_block = block_list[0]
    
    
#     prm = {}
#     for line in header_block:
# #        data_block_list = _get_data_block_list(block_list)
#         if pixel_number == None:
#             pixel_number = _count_pixelNumber(
#                 mla_txt_fn, mode=mode, verbose=verbose
#             )
#         prm['pixelNumber'] = pixel_number
#         if "nsamples" in line:
#             prm['nsamples'] = int(line[11:17])
#         if "srate" in line:
#             prm['srate'] = int(line[8:15])
#         if "bandwidth" in line:
#             prm['df'] = int(line[24:28])
#         if "amplitude" in line:
#             prm['modamp'] = float(line[23:26])
#         if "number" in line:
#             prm['demodnum'] = int(line[37:39])
#         if "offset" in line:
#             prm['offset'] = float(line[9:14])
            
#     prm['demod_freqs'] = _parse_mla_demod_frequencies(header_block)
    
#     if len(prm) == 0:
#         raise MLAReconstructionException("""Given block_list does not have a 
#                                          header block. parsing failed"""
#         )
#     return prm


def _load_mla_data_header(f, pixel_number):
    """reads the parameter value from file f and returns returns them in a
    dictionary
    
    This function uses the ``_read_one_block(..)`` function to read parameter
    values from the given file. The file can be given as a file hanlde (i.e. 
    an opened ``_io.TextIOWrapper`` object) or as a filename.
    
    Parameter
    ---------
        f | file handle 
            file handler of the txt file from which the paramter values should
            be read back. 
        pixel_number | int
            number of spectra stored in the MLA raw data file

    Returns
    -------
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``. 
    
    Raises
    ------
        MLAReconstructionException
            if the given filename does not lead to a valid file or if the 
            given file is neither of type ``str`` nor ``_io.TextIOWrapper``.
    
    """
    import _io
    
    if isinstance(f, str):
        try:
            f = open(f, mode='r')
        except FileNotFoundError:
            raise MLAReconstructionException(
                """FileNotFoundError occured. Given str: {} could not be opened.
                """.format(f)
            )
    elif not isinstance(f, _io.TextIOWrapper):
        raise TypeError(
            """Given variable f: {} is not a readable file."""
        )
    
    
    hd_block_lines = _read_one_block(f)
    block_list = _create_block_list(hd_block_lines)
    prm = _parse_mla_data_header(block_list, pixel_number=pixel_number)
    return prm


def _parse_mla_demod_frequencies(header_block):
    """returns 1d-array with demodulation frequencies parsed from mla data 
    header block
    
    This function parses the demodulation frequencies from the given 
    ``header_block`` and returns them as a 1d np.array. This function is 
    normally only used in combination with ``_load_mla_data(..)`` and 
    ``_create_block_list(..)``.
    
    
    Parameter
    ---------
        header_block | list
            list of strings where every string corresponds to a line of the header
            section from the data file. 
            
    Returns
    -------
        np.array
            contains the modulation frequencies.
            
    Example
    -------
        >>> lines = _load_mla_data('mla_data.txt')
        >>> block_list = _create_block_list(lines)
        >>> demod_freqs = _parse_mla_demod_frequencies(block_list[0])
        
        

    """
    demod_freq_idx = header_block.index(' demodulating frequencies:\n')
    
    val_list = [
        s.replace('\n','').split('  ') for s in header_block[demod_freq_idx+1:]
    ]
    arr = np.array(val_list, dtype=np.float)
    return arr.reshape((arr.size,))
    
def _get_data_block_list(block_list):
    """returns a list of data block extracted from the given block list 
    
    Selects the data blocks from the given ``block_list`` by comparing the
    first line of each block with the ``DATA_BLOCK_HEADER`` variable. Function
    ignores empty blocks.
    
    
    Parameter
    ---------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
            
    Return
    ------
        data_block_list | list
            list where every element corresponds to a data block. 

    """
    data_block_list = []
    for block in block_list:
        if len(block) > 0:
            if DATA_BLOCK_HEADER in block[0]:
                data_block_list.append(block)
    return data_block_list
 #   return [block for block in block_list if DATA_BLOCK_HEADER in block[0]]
    
def _parse_data_block(data_block):
    """returns the FFT coefficients from one data block as 1d np.array of 
    complex values.
    
    Parameter
    ---------
        data_block | list
            list of strings where every string corresponds to a line of a
            data_block. 
            
    Return
    ------
        1d np.array
            complex-valued FFT coefficient from one MLA spectrum (i.e. data block)
        dc_value | float (optional)
            MLA data block dc_value pixel. only returned if it is present in 
            the parsed datablock
            
    Example
    -------
        reconstruct one specta (without dc-value)
        
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> one_spectrum_dset = _parse_data_block(block_list[1])
    
    
    
    """
    data_block_body = []
    dc_value = None
    for line in data_block:
        if 'Part' in line or '# of trigger:' in line:   # ignore header line
            pass
        elif 'Dc Value:' in line:                       # parse dc-value
            dc_value = _parse_dc_value_from_line(line)
        else:                                           # read data line
            data_block_body.append(line)
    
    vals = np.array(
        [l.replace('\n','').split('\t') for l in data_block_body], 
        dtype=np.float
    )
    data = np.dot(vals, [1, 1j])
    
    if dc_value == None:    # normal caes
        return data
    else:                   # dc-value case
        return data, dc_value      

def _parse_all_data_blocks(block_list, demodnum, remove_compensation_channel=True):
    """returns a 2d-array of FFT coefficients from one MLA dataset
    
    Compiles a complex-valued 2d array with the FFT coefficients from one MLA 
    measurement by calling the ``_parse_data_block(..)`` function on every 
    data block in ``block_list``. 
    
    Parameter
    ---------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
        demodnum | int
            specifies the number of demodulation frequencies. Variable is 
            defined in the header block and can be imported with the 
            ``_parse_mla_data_header(..)`` function.
        remove_compensation_channel | bool
            specifies if the compensation channel (normally harmonics #32) 
            should be removed from every set of FFT coefficients.
    
    Returns
    -------
        dset | 2d np.array
            contains the FFT coefficients from all spectra in the MLA measurement.
            
    
    Example
    -------
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        >>> dset = _parse_all_data_blocks(block_list, prm['demodnum'])
    
        to select the FFT coefficents for a given spectrum
        
        >>> spectrum_idx = 12
        >>> fftcoeffs = dset[spectrum_idx, :]
    
    
    """
    # retrieve data blocks
    data_block_list = _get_data_block_list(block_list)
    
    # parse data from txt to np.array 
    dset = np.empty((len(data_block_list), demodnum), dtype=np.complex)
    dc_value = None
    for idx,data_block in enumerate(data_block_list):
        rtrn = _parse_data_block(data_block)
        if type(rtrn) != tuple:          # normal case -- one return arg
            dset[idx,:] = rtrn
        else:                            # dc value case -- multiple return args
            if len(rtrn) == 2:
                dset[idx,:], dc_value = rtrn
            else:
                raise MLAReconstructionException(
                    'Data block parsing failed. More than two return values provided.'
                )
        
    # remove compensation channel from dset
    if remove_compensation_channel:
        dset = _delete_last_fft_tone(dset)

    # case handling if dc_value is present
    if dc_value != None:
        return dset, dc_value
    else:
        return dset


def _parse_dc_value_from_line(line, dc_value_identifier= None):
    """parses the dc value from the given line 
    
    Parameter
    ---------
        line | str
            one line parsed from the mla data file containing the dc value 
        
    Returns
    -------
        dc_value | float
            parsed dc value
        
    Example:
        >>> dc_value_identifier = 'Dc Value:'
        >>> line = 'Dc Value: 2.834523'
        >>> dv_value = _parse_dc_value_from_line(
        ...     line,
        ...     dc_value_identifier
        .. )
        
        
    """
    DC_VALUE_IDENTIFIER = 'Dc Value:'
    
    if dc_value_identifier == None:
        dc_value_identifier = DC_VALUE_IDENTIFIER
        
    return float(line.replace(dc_value_identifier,''))
    
# ===========================================================================
# data processing -- add phase and amplitude lag
# ===========================================================================

def _delete_last_fft_tone(dset):
    """returns the dset with the last column deleted.
    
    This function is used to remove the fft coefficient of the compensation 
    channel from the mla data set
    """
    return np.delete(dset, -1, axis=1)


def _convert_to_polar_coordinates(dset):
    """convert the complex values in the given 2d-array into polar coordinate 
    representation
    
    For conversion the ``cmath.polar`` function is used. The return angular
    values are given in radiant.
    This function adds a third axis to the given dset array to store the 
    computed radial coordinate values as ``dset[:,:,0]`` and the angular 
    coordinate values as ``dset[:,:,1]`` .
    
    Parameter
    ---------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).
            
    Returns
    -------
        dset_p | 3d np.array
            FFT coefficient matrix in polar coordinates. The third
            axes has length two where 0 is the radial part and 1 stands for 
            the angular part (``dset_p[:,:,0]`` returns a 2d array of all 
            radial coordinate values.)
            
    Example
    -------
        load data
        
        >>> lines = _load_mla_data('mla_data.txt')
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        >>> dset = _parse_all_data_blocks(block_list, prm['demodnum'])
        
        convert complex-valued ``dset`` to polar coordinates
        
        >>> dset_p = _convert_to_polar_coordiates(dset)
        >>> len(dset_p.shape)
        ... 3
        
    Raises
    ------
        MLAReconstructionException
            if the given ``dset`` np.array is not 2-dimensional.
            
    
    """
    
    # verify that dset shape matches
    if len(dset.shape) != 2:
        err_msg = """dset must be a 2d np.array. 
            Given dset has shape: {:d}""".format(len(dset.shape))
        raise MLAReconstructionException(err_msg)
    
    # allocate memory
    dset_p = np.empty((*dset.shape,2), dtype=np.float)

    # convert radial and angular coordinates 
    for idx in range(dset.shape[0]):
        for idy in range(dset.shape[1]):
            dset_p[idx, idy,:] = cmath.polar(dset[idx, idy])
    return dset_p
    
    
def _add_amplitude_and_phase_correction(dset_p, deviation=None, phase_lag=None, 
                                        amplitude_lag=None):
    """applies amplitude and phase correction to the given FFT coefficent matrix
    
    This function applies the predefined ``amplitude_lag`` and ``phase_lag``
    values as correction onto the provided FFT coefficient matrix.
    It is important that 
        * ``dset_p`` is provided in **polar coordinates** and 
        * the ``phase_lag`` are provided in **degrees**.
    
    If no values for ``deviation``, ``phase_lag``, and ``amplitude_lag`` are 
    provided default parameter will be used (see ``DEVIATIOM``, ``PHASE_LAG``, 
    and ``AMPLITUDE_LAG`` variables defined in this file).
    
    
    Parameter
    ---------
        dset_p | 3d np.array
            FFT coefficient matrix in polar coordinates. The third
            axes has length two where 0 is the radial part and 1 stands for 
            the angular part (``dset_p[:,:,0]`` returns a 2d array of all 
            radial coordinate values).
        deviation | float
            constant offset applied to the phase lag values
        phase_lag | 1d np.array
            phase lags values for every higher harmonic. Length has to match 
            the FFT-coefficient length in ``dset_p``.
        amplitude_lag | 1d np.array
            ampltiude lag values for every higher harmonic. Length has to match
            the FFT-coefficient length in ``dset_p``.
            
    
    Returns
    -------
        3d np.array
            phase- and amplitude-corrected FFT coefficient matrix in polar
            coordinates.
        deviation | float
            constant offset applied to the phase lag values
        phase_lag | 1d np.array
            phase lags values for every higher harmonic. Length has to match 
            the FFT-coefficient length in ``dset_p``. 
        amplitude_lag | 1d np.array
            ampltiude lag values for every higher harmonic. Length has to match
            the FFT-coefficient length in ``dset_p``.    
            
            
    """
    
    # create copy of data structure
    dset_p_ = dset_p.copy()
    
    # use default correction values if phase and amplitude lag is not provided
    if deviation is None:
        deviation = DEVIATION
    if phase_lag is None:
        phase_lag = PHASE_LAG
    if amplitude_lag is None:
        amplitude_lag = AMPLITUDE_LAG
        
    # apply correction to 
    for idx_pxl in range(dset_p_.shape[0]):
        dset_p_[idx_pxl, :, 0] *= amplitude_lag
        dset_p_[idx_pxl, :, 1] += (phase_lag + deviation) * np.pi/180
        
    return dset_p_, deviation, phase_lag, amplitude_lag
    
def _convert_to_rectangular_coordinates(dset_p):
    """converts the given array from polar to cartesian coordinates (i.e. complex
    value representation).
    
    Function uses ``cmath.rect`` to convert from polar coordinates to
    rectangular coordinates. It is important that the angular input valur is 
    given in **radiant**.
    
    Parameter
    ---------
        dset_p | 3d np.array
            FFT coefficient matrix in polar coordinates. The third
            axes has length two where 0 is the radial part and 1 stands for 
            the angular part (``dset_p[:,:,0]`` returns a 2d array of all 
            radial coordinate values).

    Return
    ------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).            
    
    """
    dset_rect = np.empty(dset_p.shape[:2], dtype=np.complex)
    
    for idx_pxl in range(dset_p.shape[0]):
        for idx_fftcoeff in range(dset_p.shape[1]):
            dset_rect[idx_pxl, idx_fftcoeff] = cmath.rect(
                *dset_p[idx_pxl, idx_fftcoeff, :]
            )
    
    return dset_rect


# ===========================================================================
# data processing -- convert to energy values
# ===========================================================================
    
def _setup_reconstruction_parameter(pixelNumber, nsamples, srate, df, modamp, 
                                    demodnum, offset, demod_freqs):
    """creates and returns parameter required for the energy spectrum 
    reconstruction 
    
    This function calculates the parameters needed for the energy spectrum 
    reconstruction. input parameter are the information form the mla data 
    header.
    
    Parameter
    ---------
        pixelNumber | int
            number of spectra in the MLA dataset. This values is equal to the 
            number of data blocks in the MLA raw data file.
        nsamples | int
            number of measured data points per spectra
        srate | int
            sampling rate during the measurement.
        df | float
            frequency different between subsequent tones
        modamp | float
            modulation amplitude in Volts.
        demodnum | int
            number of tones from which spectra are reconstructed
        offset | float
            DC component of excitation signal in Volts.
        demod_freqs | 1d np.array
            frequency values of the tones used to reconstruct the spectra
            
    Returns
    -------
        t | 1d np.array
            vector of time value for one excitaiton period
        v_t | 1d np.array
            vector of Volt values for one excitation period
        first_index | int
            index of the first harmonics (i.e. base tone)
        last_index | int
            index of the last harmonics
        step_size | float
            difference between two subsequent tones in parts of ``df``.
    
    Example:
    --------
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        >>> t, v_t, first_index, last_index, step_size = _setup_reconstruction_parameter(**prm)

    """
    t = np.arange(nsamples) / srate * demod_freqs[0]
    v_t = offset + modamp * np.cos(2 * np.pi * t)
    
    first_index = int(demod_freqs[0] / df)
    last_index = int(demod_freqs[30] / df)
    step_size = int((demod_freqs[1] - demod_freqs[0]) / df)
    
    return t, v_t, first_index, last_index, step_size

def _get_excitation_time_values(prm):
    """ """
    return np.arange(prm['nsamples']) / prm['srate'] * prm['demod_freqs'][0]

def _get_excitation_voltage_signal(t, prm):
    """ """
    return prm['offset'] + prm['modamp'] * np.cos(2 * np.pi * t)

def _adjust_reconstruction_idcs_to_dc_value(dset_row, first_index, last_index, 
                                            step_size, verbose=False):
    """reduces first_indexby one  if dc_value is present in given fft coeff row.
    
    Doesn't change first_index if dc_value is not present. Presence of dc_value
    is determined by comparing the length of dset_row and the values of 
    first_index, last_index, and step_size.
    
    Parameter
    ---------
        dset_row | 1d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).           
        first_index | int
            index of the first harmonics (i.e. base tone)
        last_index | int
            index of the last harmonics
        step_size | float
            difference between two subsequent tones in parts of ``df``.

    Returns
    ---------
        first_index | int
            adjusted index of the first harmonics (i.e. base tone)
        last_index | int
            index of the last harmonics
        step_size | float
            difference between two subsequent tones in parts of ``df``.

    
    """
    row_len = dset_row.size
    idx_width = last_index + step_size - first_index
    
    if idx_width == row_len:
        pass
    elif idx_width+1 == row_len:
        first_index -= 1
        if verbose:
            print(
                'dc-value detected. shifted first_index by -1 to be {0:d}.'.format(first_index)
            )
    else:
        raise MLAReconstructionException(
            'Given row length does not match specified first_index and last_index !',
            '\n row length: {0:d}; first_index: {1:d}, last_index: {2:d}'.format(
                row_len, first_index, last_index    
            )
        )
    
    if verbose:
        print('row length: ', row_len)
        print('idcs:       ', idx_width)
    return first_index, last_index



def reconstruct_energy_spectra(
        dset, prm, t, v_t, first_index, last_index, step_size, use_trace='bwd',
        e_res=0.005
    ):
    """returns the reconstructed current, conductance data with the corresponding
    energy values.
    
    This functions generates the current and conductance values from the given
    FFT coefficient matrix and corresponding measurement parameter. This here 
    is the main function to reconstruct the MLA measurement. 
    
    
    Parameter
    ---------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored as complex values (i.e. cartesian representation).           
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        t | 1d np.array
            vector of time value for one excitaiton period
        v_t | 1d np.array
            vector of Volt values for one excitation period
        first_index | int
            index of the first harmonics (i.e. base tone)
        last_index | int
            index of the last harmonics
        step_size | float
            difference between two subsequent tones in parts of ``df``.
        use_trace | (``fwd`` | ``bwd`` | ``both``)
            indicates which part of the reconstructed spectrum will be used
            to construct the ``curr`` and ``cond`` matrices. Valid values are
            ``fwd``: use the forward trace; ``bwd``: use the backward trace; 
            or ``both``: use the average of fwd and bwd trace.
        e_res | float
            energy resolution (in volts) specifies the spacing between 
            subsequent values in the returned linearized energy np.array


    Returns
    -------
    linearizedEnergy | 1d np.array
        energy values at which the current and conductance is reconstructed
    cond | 2d np.array
        matrix of conductance values. 
    curr | 2d np.array
        matrix of current values.
    
    Example
    -------
        Select MLA raw data file

        >>> mla_data_fn = 'Measurement of 2021-03-15 2005.txt'
        
        Load MLA data and measurement parameter from txt file
        
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        
        Parse data into np.array
        
        >>> dset = get_measurement_data(block_list, prm)

        generate the conductance and current matrices 

        >>> linE, linarr, arr = get_energy_spectra(dset, prm)
    
    
    """
    
#    e_res = 0.005   # [V]
    linearizedEnergy = get_linearized_energy(prm, e_res=e_res)
    e_nr = len(linearizedEnergy)
    

    curr = np.zeros([dset.shape[0], e_nr], dtype='float64')
    cond = np.empty([dset.shape[0], e_nr], dtype='float64')
    
    for idx, row in enumerate(dset):
        
        # ======
        # FEATURE: DC-VALUE
        # ======
        first_index, las_index = _adjust_reconstruction_idcs_to_dc_value(
            row,
            first_index,
            last_index,
            step_size
        )
        # ======
        # reconstruct current from FFT coefficients
        # ======
        recon_pixels_amp = _reconstruct_current_from_fft_coefficients(
            row,
            prm,
            first_index,
            last_index,
            step_size
        )
        
        # ======
        # create currrent interpolation function
        # ======
        f = _calc_current_interpolation(
            recon_pixels_amp, 
            prm, 
            v_t, 
            use_trace=use_trace
        )
        
        # ====
        # fillin curren values
        # ====
        curr[idx, :] = f(linearizedEnergy)
        
        # ====
        # fillin conductance values
        # ====
        # dE = np.diff(linearizedEnergy)[0]
        # lin_en_ = np.concatenate([
        #     linearizedEnergy-dE/2, [linearizedEnergy[-1]+dE/2]
        # ])
        # cond_row = np.diff(f(lin_en_))
        
        cond_row = _calc_conductance_from_current(f, linearizedEnergy)
        cond[idx, :] = cond_row
        
    return linearizedEnergy, cond, curr

def _reconstruct_current_from_fft_coefficients(dset_row, prm, first_index, 
                                               last_index, step_size):
    """returns the real-valued current trace reconstructed from the FFT
    coefficents in dset_row
    
    To take dv-calue into account, function compares the dset_row and 
    first_index/last_index values. if they match no dc-value is present; if 
    they mismatch by one, the first_index is shifted.
    
    Parameter
    ---------
        dset_row | 1d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).           
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        first_index | int
            index of the first harmonics (i.e. base tone)
        last_index | int
            index of the last harmonics
        step_size | float
            difference between two subsequent tones in parts of ``df``.
    
    Returns
    -------
        recon_pixels_amp | 2d np.array
            ampltiude values of the reconstracted curren trace. 
    
    """
    fft_coeff_vec = np.zeros(prm['nsamples'], dtype=np.complex)
    
    # n_val = last_index+step_size - first_index
    # if n_val == list(dset_row.shape)[-1]:
    #     pass
    # if n_val == list(dset_row.shape)[-1]-1:
    #     first_index += -1
    # else:
    #     raise ValueError(
    #         'first_index, last_index values do not match dset_rect size'
    #     )
    
    fft_coeff_vec[first_index:last_index + step_size: step_size] = dset_row
    recon_pixels = np.fft.ifft(fft_coeff_vec)
    recon_pixels_amp = 2 * np.real(recon_pixels) * 1e3
    return recon_pixels_amp

def _calc_current_interpolation(recon_pixels_amp, prm, v_t, use_trace='bwd',
                                **kwargs):
    """returns the interpolated current trace as callable function.
    
    Interpolation is done using the ``scipy.interpolate.interp1d`` function.
    
    
    Paramter
    --------
        recon_pixels_amp | 2d np.array
            ampltiude values of the reconstracted curren trace. 
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        v_t | 1d np.array
            vector of Volt values for one excitation period
        use_trace | (``fwd`` | ``bwd`` | ``both``)
            indicates which part of the reconstructed spectrum will be used
            to construct the ``curr`` and ``cond`` matrices. Valid values are
            ``fwd``: use the forward trace; ``bwd``: use the backward trace; 
            or ``both``: use the average of fwd and bwd trace.
        kwargs |
            are passed to ``scipy.interpolate.interp1d``.

    Returns
    -------
        f | scipy.interpolate.interp1d
            interpolated current trace is returned as callable obbject
            
    Example
    -------
            
    
    """
    if use_trace == 'fwd':
        f = interpolate.interp1d(
            v_t[:prm['nsamples']//2], 
            recon_pixels_amp[:prm['nsamples']//2],
            **kwargs
        )
    elif use_trace == 'bwd':
        f = interpolate.interp1d(
            v_t[prm['nsamples']//2:], 
            recon_pixels_amp[prm['nsamples']//2:],
            **kwargs
        )
    elif use_trace == 'both':
        f = interpolate.interp1d(
            v_t, 
            recon_pixels_amp,
            **kwargs
        )
    else: 
        raise ValueError(
            "Given use_trace: {} is not valid!".format(use_trace)
        )
    return f
        
def _calc_conductance_from_current(interp_curr, linearizedEnergy):
    """returns conductance values calculated from the provided interpolated 
    current function and the given linearized energy values
    
    Parameter
    ---------
        interp_curr | scipy.interpolate.interp1d
            interpolated current trace is returned as callable obbject
        linearizedEnergy | 1d np.array
            represents the energy values for which the condutance values are 
            calculated
    
    Returns
    -------
        cond_row | 1d np.array
            reconstructed conductance values calculated from the given current 
            at the specified energy values
    
    """
    dE = np.diff(linearizedEnergy)[0]
    lin_en_ = np.concatenate([
        linearizedEnergy-dE/2, [linearizedEnergy[-1]+dE/2]
    ])
    cond_row = np.diff(
        interp_curr(lin_en_)
    )
    return cond_row
    
    
    
    


def get_linearized_energy(prm, e_res=0.005):
    """returns linearized energy values calculated from the MLA modulation 
    amplitude and offset, and the given energy resolution ``e_res``
    
    Paramter
    --------
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        e_res | float
            energy resolution (in volts) specifies the spacing between 
            subsequent values in the returned linearized energy np.array
            
    Returns
    -------
        linearized_energy | 1d np.array
            linearized energy values calculated from the MLA modulation 
            amplitude and offset.
            
            
    Example
    -------
        For this example to work the ``pixel_number`` of the used MLA dataset
        ``mla_data.txt`` must be known and past to the 
        ``_load_mla_data_header(..)`` function.
        
        >>> mla_fn = 'mla_data.txt'
        >>> prm = _load_mla_data_header(
        ...    mla_fn, 
        ...    pixel_number=pixel_number
        ... )
        >>> lin_en = get_linearized_energy(prm)
            
    """
#    e_min = prm['offset'] - prm['modamp'] + e_res/2
    e_min = prm['offset'] - prm['modamp'] + e_res
    e_max = e_max = prm['offset'] + prm['modamp']
    return np.arange(e_min, e_max, e_res)
    
    

#     arr = np.zeros([dset.shape[0], prm['nsamples']], dtype='float64')
#     linarr = np.empty([dset.shape[0], e_nr], dtype='float64')

#     # linearizedEnergy = np.linspace(-0.117, 0.278, 313)

#     for idx, row in enumerate(dset):
        
#         # ======
#         # reconstruct current from FFT coefficients
#         # ======
#         fft_coeff_vec = np.zeros(prm['nsamples'], dtype=np.complex)
# #        fft_coeff_vec[int(first_index): int(last_index) + int(step_size): int(step_size)] = row[0:31:1]
#         fft_coeff_vec[first_index:last_index + step_size: step_size] = row
#         recon_pixels = np.fft.ifft(fft_coeff_vec)
#         recon_pixels_amp = 2 * np.real(recon_pixels) * 1e3
#         arr[idx, :] = recon_pixels_amp
        
#         # ======
#         # create interpolated values for the conductance computation
#         # ======
#         f = interpolate.interp1d(
#             v_t[313:625], 
#             recon_pixels_amp[313:625]
#         )
#         linarr[idx, :] = f(linearizedEnergy)

#     # =====
#     # calculate the conductance
#     # =====
# #    linarr = np.diff(linarr) * -1
#     linarr = np.diff(linarr)
    
# #    return linearizedEnergy, linarr, arr

    # ========================================================================
    # =======================================================================
            

        
    

def get_measurement_data(block_list, prm, deviation=None, phase_lag=None, 
                         amplitude_lag=None, add_dc_value=False, 
                         nanonis_current_offset=0):
    """returns the phase- and amplitude corrected measurement data as 2d array
    
    This function parses the complex frequency values from the text file 
    data blocks and phase- and amplitude corrects them. Returns a 2d np.arrray 
    
    Function ammends the dset if the called _parse_all_data_blocks function
    returns a dc_value with the parsed dset value. Therefore the returned
    array length can vary by one depending if the parsed data has a dc_value
    or not.
    
    
    Parameter
    ---------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        deviation | float
            constant offset applied to the phase lag values
        phase_lag | 1d np.array
            phase lags values for every higher harmonic. Length has to match 
            the FFT-coefficient length in ``dset_p``.
        amplitude_lag | 1d np.array
            ampltiude lag values for every higher harmonic. Length has to match
            the FFT-coefficient length in ``dset_p``.
        add_dc_value | bool
            flag to indicate if dc_value is present and should be parsed
        nanonis_current_offset | float
            current offset value set within nanonis. is required for dc_value 
            compensation


    Returns
    -------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).       
    
    Example:
    --------
        Load the data and measurement parameter
        
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        
        generate the phase- and amplitude-corrected FFT coefficient matrix
        
        >>> dset = get_measurement_data(block_list, prm)
    
    
    """
    
    # parse all data blocks
    rtrn = _parse_all_data_blocks(block_list, prm['demodnum'])
    
    # case handling if dc_value is present
    dc_value = None
    if type(rtrn) != tuple:
        dset = rtrn
    else:
        if len(rtrn) == 2:
            dset, dc_value = rtrn
        else:
            raise MLAReconstructionException(
                'Parse all data blocks failed. Invalid number of return args.'
            )
        
    # apply signal corrections
    dset_rect = apply_amplitude_and_phase_correction(
        dset, 
        prm,
        deviation=deviation,
        phase_lag=phase_lag,
        amplitude_lag=amplitude_lag
    )
    
    # add dc_value to corrected dset_rect
    if add_dc_value and dc_value != None:
        dset_rect = add_dc_value_to_fft_coeff_array(
            dset, dc_value, nanonis_current_offset
        )
    return dset_rect

def add_dc_value_to_fft_coeff_array(dset, dc_value, nanonis_current_offset):
    """returns the dset array with dc_value added as its first entry
    
    Parameter
    ---------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).       
        dc_value | float
            dc_value for mla reconstruction
        nanonis_current_offset | float
            current offset value set within nanonis. is required for dc_value 
            compensation

        
    Returns
    -------
        dset | 2d np.array
            FFT coefficient matrix from a MLA measurement. Coefficient are 
            stored complex values (i.e. cartesian representation).       
    
    """
    shp = dset.shape[0], 1
    print('                  dc-value: ', dc_value-nanonis_current_offset)
    dc_value_arr = np.ones(shp) * (dc_value - nanonis_current_offset) 
    return np.concatenate((dc_value_arr, dset), axis=1)
     
    
    
    
def apply_amplitude_and_phase_correction(dset, prm, deviation=None, phase_lag=None, 
                         amplitude_lag=None):
    """ """
    dset_p = _convert_to_polar_coordinates(dset)
    dset_p, deviation, phase_lag, amplitude_lag = _add_amplitude_and_phase_correction(
        dset_p, 
        deviation=deviation,
        phase_lag=phase_lag,
        amplitude_lag=amplitude_lag
    )
    dset_rect = _convert_to_rectangular_coordinates(dset_p)
    return dset_rect    
    

def get_energy_spectra(dset, prm, e_res=0.005):
    """ """
    t, v_t, first_index, last_index, step_size = _setup_reconstruction_parameter(**prm)

    linearizedEnergy, linarr, arr = reconstruct_energy_spectra(
        dset, prm,
        t, v_t, first_index, last_index, step_size,
        e_res=e_res
    )
    return linearizedEnergy, linarr, arr
    
    
    
def generate_energy_spectra(mla_data_fn, n=1000, e_res=0.005, verbose=True):
    """loades, processes, and saves MLA energyspectra blockwise to text file.
    
    Function loads FFT coefficients from MLA measurement file, calculates
    the energy spectra, and saves the energy values to a text file. This
    process is done in a block-wise fashion to avoid loading all the data at
    once into the memory. This function can be used to process large datasets.
    
    Parameter
    ---------
        mla_data_fn | str
            file path to the MLA raw data text file
        n | int
            number of blocks into which the MLA data will be parsed for block-
            processing
        e_res | float
            energy resolution (in volts) specifies the spacing between 
            subsequent values in the returned linearized energy np.array
        verbose | bool
            flag to printout energy-spectra generation progress
    
    
    Example:
    --------
        This code snipped shows how to generate a txt file with all energy 
        spectra.
        
        >>> mla_data_fn = 'Measurement of 2021-03-07 1432.txt'  # example data file
        >>> generate_energy_specta(mla_data_fn)
    
    """
    
    # ==================
    # define file names
    # ==================
    fn_arr = mla_data_fn.replace('.txt', '_arr.txt')
    fn_linarr = mla_data_fn.replace('.txt', '_linarr.txt')
    fn_linE = mla_data_fn.replace('.txt', '_linE.txt')

    
    # ==================
    # load block_list and parameter
    # ==================
    lines = _load_mla_data(mla_data_fn)
    block_list = _create_block_list(lines)
    prm = _parse_mla_data_header(block_list)
    data_block_list = _get_data_block_list(block_list)    
    
    
    
    n = 1000
    data_block_parcels = [
        data_block_list[i:i+n] for i in range(0, len(data_block_list), n)
    ]
    n_parcles = len(data_block_parcels)
    
    
    for idx, sub_block in enumerate(data_block_parcels):
        
        # ====================
        # load data parcel and compute energy spectra
        # ====================
        dset = get_measurement_data(sub_block, prm)
        linE, linarr, arr = get_energy_spectra(dset, prm, e_res=e_res)
        
        # ===================
        # save spectra to file
        # ===================
        f_arr = open(fn_arr, 'a')
        f_linarr = open(fn_linarr, 'a')
        np.savetxt(f_arr, arr)
        np.savetxt(f_linarr, linarr)
        f_arr.close()
        f_linarr.close()
        
        
        if verbose:
            print('converted {0:d}/{1:d}'.format(idx, n_parcles))
    
    # ================
    # save energy values
    # ================
    f_linE = open(fn_linE, 'a')
    np.savetxt(f_linE, linE)
    f_linE.close()
    
    return


# ===========================================================================
# data processing -- zero-pad missing trigger
# ===========================================================================

def compile_missing_pixel_list(missing_locations, n_z=40):
    """returns missing_pxls list compiled from a missing_locations list.
    
    The missing_pxls list contians the index of the missing measurement pixels
    for every height value for all the provided measurement locations The
    missing_locations dictionary contains only the the list of missing 
    measurement locations. 
    
    Paramter
    --------
        missing_locations | list of integer
            contains index of missed grid locations in the to-be-reconstructed MLA
            dataset. Attention, keep in mind the difference between **locations
            indices** and **pixel indices** when wokring with this list.
        n_z | int
            number of MLA measurements at every grid location. Normally this
            refers to the different heights which are measured per grid 
            location.
            
    Returns
    -------
        missing_pxls | list of integer
            specifies the missing MLA measurement point. Attention, keep in 
            mind the difference between **locations indices** and **pixel 
            indices** when wokring with this list.
    
    
    Example:
    --------
    A grid-MLA with 40 height values measured at every
    measurement location on a 20x20 (= nrows x ncols) grid misses a pixel at 
    the grid location  14x5 (=row_idx=14; col_idx=5) would use:
        
    >>> 
    >>> nrows= 20; ncols=20; n_z = 40
    >>> idx_row = 14-1; idx_col = 5-1
    >>> missing_locations = [
    ...     nrows*idx_row + idx_col
    ... ]
    >>> missing_pxls = compile_missing_pixel_list(
    ...     missing_locations, n_z=n_z
    ... )
    
    
    """
    missing_pxls = []
    
    for loc_idx in missing_locations:
        missing_pxls += list(range(loc_idx*n_z,(loc_idx+1)*n_z))
    return missing_pxls


def _zeropad_missing_pxl(data_block_list, missing_pxl_idx, zsweep_nr,
                             prm):
    """inserts n= ``zsweep_nr`` dummy data blocks into the given 
    ``data_block_list`` at the index ``missing_pxl_idx``
    
    Parameter
    --------
        data_block_list | list
            list where every element corresponds to a data block. 
        missing_pxl_idx | int
            index at which the dummy data blocks will be inserted
        zsweep_nr | int
            number of zsweeps per measurement location. This is a measurement
            parameter.
        prm | dict
            contains the measurement parameter present in the datafile header
            block. Can be generated with the ``_parse_mla_data_header(..)``.
        
    Returns
    -------
        None
        
    Attention
    ---------
        The ``missing_pxl_idx`` variable refers to the number of the physical
        measurement location, not to the spectrum idx number. The relation 
        between those to index number is given as
            >>> spectrum_idx = missing_pxl_idx * zsweep_nr
        

    Example
    -------
        Load the MLA ``data_block_list`` from text file
        
        >>> lines = _load_mla_data(mla_txt_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        >>> data_block_list = _get_data_block_list(block_list)
     
        Zero-pad the missing pixels
        
        >>> missing_pxl_idx = 52178
        >>> zsweep_nr = 40
        >>> _zeropad_missing_pxl(data_block_lost, missing_pxl, zsweep_nr, prm)
    
    """
    for dummy_db in [_create_dummy_data_block(prm) for i in range(zsweep_nr)]:
        data_block_list.insert(
            missing_pxl_idx*zsweep_nr, 
            dummy_db
        )
        data_block_list.pop(-1)
    return 


def _create_dummy_data_block(prm):
    """returns a dummy datablock with all FFT coefficients set zo zero.
    
    This function is mainly used to provide dummy data blocks for zero-padding
    data sets where missing triggers occured (missing trigger refers to 
    measurement setup issue where the MLA hardware didn't record at all 
    measurement locations).
    
    Parameter
    ---------
        prm | dict
            contains the measurement parameter present in the data file header
            block. 
    
    Returns
    -------
        dummy_data_block | list
            list of strings where every string corresponds to a line of a
            data_block. This data block has all FFT coefficient set to zero.

    Example
    -------
        Creating a dummy datablock requires to have the data parameter ready
    
        >>> lines = _load_mla_data(mla_txt_fn)
        >>> block_list = _create_block_list(lines)
        >>> prm = _parse_mla_data_header(block_list)
        >>> dummy_data_block = _create_dummy_data_block(prm)
    
        the dummy data block can be converted to dummy float values by calling
        
        >>> dummy_data_block_list = 10*[dummy_data_block]
        >>> dummy_dset = get_measurement_data(
        ...     dummy_data_block_list, 
        ...     prm
        ... )
        
    
    """
    hd = [
        '# of trigger: -1 \n',
        ' Real Part\t Imaginary Part\n',
    ]
    line = '{0:.8e}\t {1:.8e}\n'.format(0,0)
    return hd + prm['demodnum']*[line]


# ===========================================================================
# MLA dataset
# ===========================================================================

def _check_resize_curr(resize_curr, prm, e_res):
    """raises a ValueError if the given resize_curr doesn't match the dataset
    size specified in prm
    
    """
    lin_en = lin_en = get_linearized_energy(prm, e_res=e_res)
    nsamples = len(lin_en)
    
    err_msg = """ Given resize_curr: {0:} is not compatible with the loaded data:
        pixelNumber: {1:d}; and linearized-energy spacing: {2:d}""".format(
            resize_curr,
            prm['pixelNumber'],
            nsamples
    )
    
    if not prm['pixelNumber'] == np.product(resize_curr[:-1]):
        raise ValueError(err_msg)
        
    if not nsamples == resize_curr[-1]:
        raise ValueError(err_msg)
        
    return


def _check_resize_cond(resize_cond, prm, e_res):
    
    lin_en = lin_en = get_linearized_energy(prm, e_res=e_res)
    nsamples = len(lin_en)
    
    err_msg = """ Given resize_cond: {0:} is not compatible with the loaded data:
        pixelNumber: {1:d}; and linearized-energy spacing: {2:d}""".format(
            resize_cond,
            prm['pixelNumber'],
            nsamples
    )
    
    if not prm['pixelNumber'] == np.product(resize_cond[:-1]):
        raise ValueError(err_msg)
        
    if not nsamples == resize_cond[-1]:
        raise ValueError(err_msg)
        
    return


def _read_one_block(f, data_block_delimiter=None):
    """reads and returns one data block from the given file handle f
    
    Function calls ``readline()`` on the given file handle ``f`` until an 
    a ``data_block_limiter`` is read.
    
    Parameter
    ---------
        f | file handle 
            file handler of the txt file from which the data block should be 
            read back. This file handle needs to be open, otherwise the 
            function returns a ``MLAReconstructionException``.
        data_block_delimiter | list
            list of string values which are used to inidcate the end of a 
            (data) block. Default values are specified in 
            ``DATA_BLOCK_DELIMITER``.
    
    Returns
    -------
        block_lines | list
            list where every element is a line from of the read-back block.
    
    Example
    -------
        >>> fn_txt = 'Measurement of 2021-03-20 0655.txt'
        >>> f = open(fn_txt, mode='r')
        >>> block_lines = _read_one_data_block(f)
    
    Raises
    ----------
        MLAReconstructionException
            if the given file handler is not open (i.e. if ``f.closed`` is 
            ``True``).
            
    
    """
    DC_VALUE_IDENTIFIER = 'Dc Value:'
    
    if data_block_delimiter == None:
        data_block_delimiter = DATA_BLOCK_DELIMITER
    
    if f.closed:
        raise MLAReconstructionException('Given file hanlder is not open.')
    
    dc_value_line = None
    block_lines = []
    s_ = f.readline()
    while not s_ in DATA_BLOCK_DELIMITER:
        if DC_VALUE_IDENTIFIER in s_:
            dc_value_line = s_
            f.readline()
        else:
            block_lines.append(s_)
        s_ = f.readline()
        
    # insert dc value as second line 
    if dc_value_line != None:
        block_lines = block_lines[0:1] + [dc_value_line] + block_lines[1:]
    
    return block_lines


def _load_mla_data_into_hdf5(mla_data_fn, resize_curr=False, resize_cond=False, 
                             pixel_number=None, zsweep_nr=40, missing_pxls=None,
                             deviation=None, phase_lag=None, amplitude_lag=None,
                             mode='a', e_res=0.005, mla_hdf5_fn=None, 
                             add_dc_value=True, nanonis_current_offset=0,
                             verbose=True):
    """loads MLA raw data text file, computes the current and conductance maps, 
    and stores to a hdf5 file.
    
    Attention
    ---------
        It is important that the last element in ``resize_curr`` and 
        ``resize_cond`` matches whith the number of linearized-energy values,
        that will be used for the energy-spectrum reconstruction. The number
        of linearized-energy values can be deduced by calling the function
        ``get_linearized_energy(..)`` with the ``prm`` dictionary of the 
        corresponding MLA rawdataset.
    
    Parameter
    ---------
        mla_data_fn | str
            filename of the MLA raw data txt file
        resize_curr | tuple
            defines the axis and their lengths into which the computed data 
            will be  casted. This typically includes the real-spaces axes (i.e. 
            measurement frame pixel width, z-steps number, and energy vector 
            length)
        resize_cond | tuple
            defines the axis and their lengths into which the computed data will be 
            casted. This typically includes the real-spaces axes (i.e. measurement
            frame pixel width, z-steps number, and energy vector length)
        pixel_number | int
            number of spectra stored in the MLA raw data file
        zsweep_nr | int
            number of zsweeps per measurement location. This is a measurement
            parameter.
        missing_pxl_idx | int
            index at which the dummy data blocks will be inserted
        deviation | float
            constant offset applied to the phase lag values
        phase_lag | 1d np.array
            phase lags values for every higher harmonic. Length has to match 
            the FFT-coefficient length in ``dset_p``.
        amplitude_lag | 1d np.array
            ampltiude lag values for every higher harmonic. Length has to match
            the FFT-coefficient length in ``dset_p``.
        mode | str
            single characcter which specifies mode in which the HDF5 file is 
            opened in. Check the ``open(..)`` documentation for details.
        e_res | float
            energy resolution used to generate the linearized-energy vector.
        mla_hdf5_fn | str
            defines the filename of the generated HDF5 file. Default value is 
            the same name as the provided txt file with the file ending changed 
            to .hdf5
        add_dc_value | bool
            flag to indicate if dc_value is present and should be parsed
        nanonis_current_offset | float
            current offset value set within nanonis. is required for dc_value 
            compensation
        verbose | bool
            specifies if conversion progress should be plotted to the console
    
    
    Returns
    -------
        mla_hdf5_fn | str
            filename of the generated HDF5 data file
    
    Example:
    --------
        Generate the HDF5 data file from a MLA raw data text file
    
        >>> mla_txt_fn = '2021-03-15_TSP-MLA-zsweep_5x5nm_25x25pxl/Measurement of 2021-03-15 2005.txt'
        >>> mla_hdf5_fn = _load_mla_data_into_hdf5(
        ...     mla_txt_fn,
        ...     resize_curr = (25,25,40,119),    # (nrows, ncols, n_z. n_e)
        ...     resize_cond = (25,25,40,119)     # (nrows, ncols, n_z. n_e)
        ... )
    
        Load the current and conductance data from the HDF5 file back into 
        python
        
        >>> import h5py
        >>> f = h5py.File(mla_hdf5_fn, 'a')
        >>> cond = f['cond']
        >>> curr = f['curr']
        
        The conductance and current data are now loaded as multi-dimensional 
        arrays and accessible through the variables ``cond`` and ``curr``.
    
    
    
    """

    # ========================
    # set amplitude and phase lag
    # ========================    
    if deviation is None:
        deviation = DEVIATION
    if phase_lag is None:
        phase_lag = PHASE_LAG
    if amplitude_lag is None:
        amplitude_lag = AMPLITUDE_LAG

    # ========================
    # open txt file
    # ========================    
    f_txt = open(mla_data_fn, 'r')
    
    # ========================
    # extract parameter
    # ========================    
    prm = _load_mla_data_header(f_txt, pixel_number=pixel_number)
    
    # ========================
    # check if specified resize values match data length
    # ========================        
    if resize_curr != False:
        _check_resize_curr(resize_curr, prm, e_res)
    else: 
        lin_en = get_linearized_energy(prm, e_res=e_res)
        resize_curr = (prm['pixelNumber'], len(lin_en))

    if resize_cond != False:
        _check_resize_cond(resize_cond, prm, e_res)
    else: 
        lin_en = get_linearized_energy(prm, e_res=e_res)
        resize_cond = (prm['pixelNumber'], len(lin_en))
    
    
    # ========================
    # create hdf5 data structure
    # ========================
    if mla_hdf5_fn is None:
        mla_hdf5_fn = mla_data_fn.replace('.txt', '.hdf5')
        
    if os.path.exists(mla_hdf5_fn):
        os.remove(mla_hdf5_fn)
    f = h5py.File(mla_hdf5_fn, mode=mode)
    
    if add_dc_value:
        dset_len = prm['demodnum']
    else:
        dset_len = prm['demodnum']-1
    
    dset = f.create_dataset(
        'dset', 
        (prm['pixelNumber'], dset_len),
        dtype=np.complex128
    )
    curr = f.create_dataset(
        'curr', 
        resize_curr,
        dtype=np.float
    )
    cond = f.create_dataset(
        'cond', 
        resize_cond,
        dtype=np.float
    )
    
    # ========================
    # zero-pad for missing trigger
    # ========================        
    if missing_pxls == None:
        missing_pxls = []
    
    # ========================
    # load, process, and store one spectrum at a time
    # ========================
    idx_offset = 0
    
    curr_idcs = list(
        itertools.product(*[range(i) for i in resize_curr[:-1]])
    )

    for idx in range(prm['pixelNumber']):

        # ====================
        # print progress
        # ====================
        if verbose:
            if idx%10 == 0:
                print('Processed {0:d}/{1:d}'.format(idx, prm['pixelNumber']))
        
        # ====================
        # read one data block
        # ====================
        if idx in missing_pxls:
            block_lines = _create_dummy_data_block(prm)
            idx_offset += 1
        else:
            block_lines = _read_one_block(f_txt)
                    
        # ====================
        # parse block lines into data blocks
        # ====================
        block_list = _create_block_list(block_lines)
        data_blocks = _get_data_block_list(block_list)
    
        # ====================
        # reconstruct spectra & populate hdf5 datasets
        # ====================    
        if len(data_blocks) > 0:
            arr_ = get_measurement_data(
                data_blocks, 
                prm,
                deviation=deviation,
                phase_lag=phase_lag,
                amplitude_lag=amplitude_lag,
                add_dc_value=add_dc_value,
                nanonis_current_offset=nanonis_current_offset
            )
            dset[idx] = arr_
        
            linE, cond_arr, curr_arr = get_energy_spectra(
                arr_,
                prm,
                e_res=e_res
            )
            
            curr[curr_idcs[idx]] = curr_arr
            cond[curr_idcs[idx]] = cond_arr
    
    # ========================
    # save linearized energy values
    # ========================    
    lin_en = f.create_dataset(
        'lin_en', 
        linE.shape,
        maxshape = (None,),
        chunks=True,
        dtype=np.float
    )
    lin_en[:] = linE[:]
        
    # ========================
    # store measurement prm as dset attributes
    # ========================
    for k,v in prm.items():
        dset.attrs[k] = v
        
    # ========================
    # store amplitude- and phase lag calibration values
    # ========================
    mla_calibration = {
        'amplitude_lag': amplitude_lag,
        'phase_lag':     phase_lag,
        'deviation':     deviation
    }
    
    mla_calib = f.create_dataset(
        'mla_calibration', 
        resize_cond,
        dtype=np.float
    )
    for k,v in mla_calibration.items():
        mla_calib.attrs[k] = v
    
    # ========================
    # close hdf5 and txt file
    # ========================        
    f.close()
    f_txt.close()
    
    if verbose:
        print('data are successfully converted to the hdff file: {}'.format(
            mla_hdf5_fn
        ))
    
    return mla_hdf5_fn




def _load_mla_data_into_hdf5_old(mla_data_fn, resize_curr=False, resize_cond=False, 
                             pixel_number=None, zsweep_nr=40, missing_pxls=None,
                             deviation=None, phase_lag=None, amplitude_lag=None,
                             mode='a', verbose=False):
    """loads MLA raw data text file, computes the current and conductance maps, 
    and stores to a hdf5 file.
    
    Attention
    ---------
        It is important that the last element in ``resize_curr`` and 
        ``resize_cond`` matches whith the number of linearized-energy values,
        that will be used for the energy-spectrum reconstruction. The number
        of linearized-energy values can be deduced by calling the function
        ``get_linearized_energy(..)`` with the ``prm`` dictionary of the 
        corresponding MLA rawdataset.
    
    Parameter
    ---------
        mla_data_fn | str
            filename of the MLA raw data txt file
        resize_curr | tuple
            defines the axis and their lengths into which the computed data 
            will be  casted. This typically includes the real-spaces axes (i.e. 
            measurement frame pixel width, z-steps number, and energy vector 
            length)
        resize_cond | tuple
            defines the axis and their lengths into which the computed data will be 
            casted. This typically includes the real-spaces axes (i.e. measurement
            frame pixel width, z-steps number, and energy vector length)
        pixel_number | int
            number of spectra stored in the MLA raw data file
        zsweep_nr | int
            number of zsweeps per measurement location. This is a measurement
            parameter.
        missing_pxl_idx | int
            index at which the dummy data blocks will be inserted
        deviation | float
            constant offset applied to the phase lag values
        phase_lag | 1d np.array
            phase lags values for every higher harmonic. Length has to match 
            the FFT-coefficient length in ``dset_p``.
        amplitude_lag | 1d np.array
            ampltiude lag values for every higher harmonic. Length has to match
            the FFT-coefficient length in ``dset_p``.
        mode | str
            single characcter which specifies mode in which the HDF5 file is 
            opened in. Check the ``open(..)`` documentation for details.
        verbose | bool
            specifies if conversion progress should be plotted to the console
    
    
    Returns
    -------
        mla_hdf5_fn | str
            filename of the generated HDF5 data file
    
    Example:
    --------
        Generate the HDF5 data file from a MLA raw data text file
    
        >>> mla_txt_fn = '2021-03-15_TSP-MLA-zsweep_5x5nm_25x25pxl/Measurement of 2021-03-15 2005.txt'
        >>> mla_hdf5_fn = _load_mla_data_into_hdf5(
        ...     mla_txt_fn,
        ...     resize_curr = (40,25,25,625),
        ...     resize_cond = (40,25,25,322)
        ... )
    
        Load the current and conductance data from the HDF5 file back into 
        python
        
        >>> import h5py
        >>> f = h5py.File(mla_hdf5_fn, 'a')
        >>> cond = f['cond']
        >>> curr = f['curr']
        
        The conductance and current data are now loaded as multi-dimensional 
        arrays and accessible through the variables ``cond`` and ``curr``.
    
    
    
    """
    e_res = 0.005
    
    
    # ========================
    # set amplitude and phase lag
    # ========================    
    if deviation is None:
        deviation = DEVIATION
    if phase_lag is None:
        phase_lag = PHASE_LAG
    if amplitude_lag is None:
        amplitude_lag = AMPLITUDE_LAG

    
    # ========================
    # load measurement parameter
    # ========================    
    prm = _load_mla_data_header(mla_data_fn, pixel_number=pixel_number)
    
    
    # ========================
    # check if specified arr size matches data length
    # ========================        
    if resize_curr != False:
        _check_resize_curr(resize_curr, prm, e_res)
    else: 
        lin_en = get_linearized_energy(prm, e_res=e_res)
        resize_curr = (prm['pixelNumber'], len(lin_en))

    if resize_cond != False:
        _check_resize_cond(resize_cond, prm, e_res)
    else: 
        lin_en = get_linearized_energy(prm, e_res=e_res)
        resize_cond = (prm['pixelNumber'], len(lin_en))
    
    
    # ========================
    # create hdf5 data structure
    # ========================
    mla_hdf5_fn = mla_data_fn.replace('.txt', '.hdf5')
    f = h5py.File(mla_hdf5_fn, mode=mode)
    
    dset = f.create_dataset(
        'dset', 
        (prm['pixelNumber'], prm['demodnum']-1),
        dtype=np.complex128
    )
    curr = f.create_dataset(
        'curr', 
        resize_curr,
        dtype=np.float
    )
    cond = f.create_dataset(
        'cond', 
        resize_cond,
        dtype=np.float
    )
    
    # ========================
    # load measurement parameter
    # ========================    

    if verbose: print('load mla data from text file')
    
    lines = _load_mla_data(mla_data_fn)
    block_list = _create_block_list(lines)
    
    data_block_list = _get_data_block_list(block_list)    

    if verbose: print('\t ...completed.\n')

    # ========================
    # zero-pad for missing trigger
    # ========================        
    if missing_pxls == None:
        missing_pxls = []
    
    for missing_pxl_idx in missing_pxls:
        _zeropad_missing_pxl(
            data_block_list, 
            missing_pxl_idx, 
            zsweep_nr, 
            prm
        )

    # ========================
    # load fft coefficients into hdf5
    # ========================
    n = 1000
    data_block_parcels = [
        data_block_list[i:i+n] for i in range(0, len(data_block_list), n)
    ]
    n_parcels = len(data_block_parcels)


    for idx, sub_block in enumerate(data_block_parcels):
        arr = get_measurement_data(
            sub_block, 
            prm,
            deviation=deviation,
            phase_lag=phase_lag,
            amplitude_lag=amplitude_lag
        )
        dset[idx*n:(idx+1)*n] = arr
        
        print('dset - converted {0:d}/{1:d}'.format(idx, n_parcels))
        
        
    # ========================
    # calculate energy spectra
    # ========================
    curr_idcs = itertools.product(*[range(i) for i in resize_curr[:-1]])
    for dset_idx, curr_idx in enumerate(curr_idcs):
        linE, cond_arr, curr_arr = get_energy_spectra(
            dset[dset_idx:dset_idx+1], 
            prm
        )
        cond[curr_idx] = cond_arr
        curr[curr_idx] = curr_arr

        print('cond - converted {0:d}/{1:d}'.format(dset_idx, prm['pixelNumber']))

    
    # ========================
    # save linearized energy values
    # ========================    
    lin_en = f.create_dataset(
        'lin_en', 
        linE.shape,
        maxshape = (None,),
        chunks=True,
        dtype=np.float
    )
    lin_en[:] = linE[:]
    
    
    # ========================
    # store measurement prm as dset attributes
    # ========================
    for k,v in prm.items():
        dset.attrs[k] = v
        
    # ========================
    # store amplitude- and phase lag calibration values
    # ========================
    mla_calibration = {
        'amplitude_lag': amplitude_lag,
        'phase_lag':     phase_lag,
        'deviation':     deviation
    }
    
    mla_calib = f.create_dataset(
        'mla_calibration', 
        resize_cond,
        dtype=np.float
    )
    for k,v in mla_calibration.items():
        mla_calib.attrs[k] = v
    

    # ========================
    # close hdf5 file
    # ========================        
    f.close()
    
    return mla_hdf5_fn


def _generate_curr_and_cond(resize_curr, dset, prm, curr, cond):
    
    curr_idcs = itertools.product(*[range(i) for i in resize_curr[:-1]])
    for dset_idx, curr_idx in enumerate(curr_idcs):
        linE, cond_arr, curr_arr = get_energy_spectra(
            dset[dset_idx:dset_idx+1], 
            prm
        )
        cond[curr_idx] = cond_arr
        curr[curr_idx] = curr_arr

        print('cond - converted {0:d}/{1:d}'.format(dset_idx, prm['pixelNumber']))
  
    


def _calculate_qpi_maps(mla_hdf5_fn):
    """
    
        ** not complete! **
    
    """

    
    # ========================
    # create qpi data structure
    # ========================    
    f = h5py.File(mla_hdf5_fn, 'r')
    cond = f['cond']

    mla_qpi_fn = mla_hdf5_fn.replace('.hdf5', '_qpi.hdf5')
    f_qpi = h5py.File(mla_qpi_fn, 'a')

    qpi = f_qpi.create_dataset(
        'qpi',
        cond.shape,
        dtype=np.complex
    )
    
    
    
    # ========================
    # create qpi data structure
    # ========================    
    f.close()
    f_qpi.close()
    
    
    return mla_qpi_fn
    

# ===========================================================================
# MLA dataset
# ===========================================================================

class MLADataset():
    """ """
    
    def __init__(self, mla_data_fn):
        
        self._init_data(mla_data_fn)
        
    def _init_data(self, mla_data_fn):
        """ """
        lines = _load_mla_data(mla_data_fn)
        block_list = _create_block_list(lines)
        
        # add measurement parameter as object attributes
        self._add_measurement_parameter(block_list)
        
        # add dataset as object attribute
        self._add_measurement_data(block_list)
        
        # reconstruct energy spectra
        self.reconstruct_energy_spectra()
        
    def _add_measurement_parameter(self, block_list):
        prm = _parse_mla_data_header(block_list)
        setattr(self, 'prm', prm)

    def _add_measurement_data(self, block_list):
        dset_rect = get_measurement_data(block_list, self.prm)
        setattr(self, 'dset', dset_rect)

    def reconstruct_energy_spectra(self):
        linearizedEnergy, linarr, arr = get_energy_spectra(self.dset, self.prm)

        setattr(self, 'linearized_energy', linearizedEnergy)
        setattr(self, 'linarr', linarr)
        setattr(self, 'arr', arr)
#




if __name__ == "__main__":
    
    # # ========================
    # # load data
    # # ========================
    # repo_dir = os.path.dirname(os.getcwd())
    # repo_dir = '/home/kh/code/wte2-bulk-qpi'
    # data_dir = 'data'
    # data_subdir = '2021-03-07_TSP-MLA_200k_450x450nm'
    
    # mla_data_fn = os.path.join(
    #     repo_dir,
    #     data_dir,
    #     data_subdir,
    #     'Measurement of 2021-03-07 1432.txt'
    # )

    # # ================================
    # # Example: reconstruct spectra using functions
    # # ================================
    # lines = _load_mla_data(mla_data_fn)
    # block_list = _create_block_list(lines)
    # prm = _parse_mla_data_header(block_list)
    # dset = _parse_all_data_blocks(block_list, prm['demodnum'])  # obsolete
    # dset_p = _convert_to_polar_coordinates(dset)
    # dset_p, deviation, phase_lag, amplitude_lag = _add_amplitude_and_phase_correction(dset_p)
    # dset_rect = _convert_to_rectangular_coordinates(dset_p)
    # t, v_t, first_index, last_index, step_size = _setup_reconstruction_parameter(**prm)
    
    # linearizedEnergy, linarr, arr = reconstruct_energy_spectra(
    #     dset_rect, prm,
    #     t, v_t, first_index, last_index, step_size,
    # )

    # # ===============================
    # # Example: reconstruct spectra using MLA object
    # # ===============================
    # mla = MLADataset(mla_data_fn)
    
    pass
