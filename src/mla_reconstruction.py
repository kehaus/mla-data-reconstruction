#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:15:39 2021

@author: kh
"""


__author__ = "kha"
__version__ = "0.0.2"


import os 
import sys
import cmath
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import h5py


# ===========================================================================
# static MLA compensation values
# ===========================================================================
DEVIATION = 0
PHASE_LAG = np.array([
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

# ===========================================================================
# load and parse functions
# ===========================================================================
class MLAReconstructionException():
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
    
    Function parses the provided list of lines and groups them into blocks and 
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
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
    
    
    """
    block_list = [[]]

    for line in lines:
        if line != '\n':
            block_list[-1].append(line)
        else:
            block_list.append([])
    return block_list

def _parse_mla_data_header(block_list):
    """returns a dictionary with the measurement parameters parsed from the 
    given header_block
    
    This funtions parses measurement parameter like ``pixelNumber``, ``nsamples``,
    and ``demod_freqs`` from the header block in the provided block list.
    The header block is selected as the first element in the block list. 
    it is important that this is actually the header block otherwise this 
    function will return an empty dictionary.
    
    Parameter
    ---------
        block_list | list
            List where every element corresponds to a block (header, data, ...).
            One block (i.e. element) consists of a list of lines belonging
            to that block of data.
            
    Returns
    -------
        prm | dict
            contains the measurement parameter present in the datafile header
            block. 
    
    Example
    -------
        >>> lines = _load_mla_data('mla_data.txt')
        >>> block_list = _create_block_list(lines)
        >>> hd_prm = _parse_mla_data_header(block_list)
    
    Raises
    ------
        MLAReconstructionException
            if given block_list does not contain a header block
    
    """
    header_block = block_list[0]
    prm = {}
    for line in header_block:
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
        raise MLAReconstructionException("""Given block_list does not have a header block. parsing failed""")
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
    first line of each block with the ``DATA_BLOCK_HEADER`` variable
    
    
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
    return [block for block in block_list if DATA_BLOCK_HEADER in block[0]]
    
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
    
    
    Example
    -------
        reconstruct one specta
        
        >>> lines = _load_mla_data(mla_data_fn)
        >>> block_list = _create_block_list(lines)
        >>> one_spectrum_dset = _parse_data_block(block_list[1])
    
    
    
    """
    vals = np.array(
        [l.replace('\n','').split('\t') for l in data_block[2:]], 
        dtype=np.float
    )
    data = np.dot(vals, [1, 1j])
    return data

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
        >>> prm = _parse_mla_data_header(block_list[0])
        >>> dset = _parse_all_data_blocks(block_list, prm['demodnum'])
    
        to select the FFT coefficents for a given spectrum
        
        >>> spectrum_idx = 12
        >>> fftcoeffs = dset[spectrum_idx, :]
    
    
    """
    
    # retrieve data blocks
    data_block_list = _get_data_block_list(block_list)
    
    # parse data from txt to np.array 
    dset = np.empty((len(data_block_list), demodnum), dtype=np.complex)
    for idx,data_block in enumerate(data_block_list):
        dset[idx,:] = _parse_data_block(data_block)
        
    # remove compensation channel from dset
    if remove_compensation_channel:
        dset = _delete_last_fft_tone(dset)
    return dset
    

# ===========================================================================
# data processing -- add phase and amplitude lag
# ===========================================================================

def _delete_last_fft_tone(dset):
    """returns the dset with the last column deleted.
    
    This function is used to remove the fft coefficient of the compensation channel from the mla data set
    
    """
    return np.delete(dset, -1, axis=1)


def _convert_to_polar_coordinates(dset):
    """converts the complex values in the given 2d-array into polar coordinate representation
    
    the return array has 3 axes where the third axis is of size two such that dset_p[:,:,0] gives the radial contribution and
    dset_p[:,:,1] returns the angular contribution of the polar coordinate tuple.
    
    """
    dset_p = np.empty((*dset.shape,2), dtype=np.float)

    for idx in range(dset.shape[0]):
        for idy in range(dset.shape[1]):
            dset_p[idx, idy,:] = cmath.polar(dset[idx, idy])
    return dset_p
    
    
def _add_amplitude_and_phase_correction(dset_p, deviation=None, phase_lag=None, amplitude_lag=None):
    """ """
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
#        dset_p_[idx_pxl, :, 0] += amplitude_lag
        dset_p_[idx_pxl, :, 0] *= amplitude_lag
        dset_p_[idx_pxl, :, 1] += (phase_lag + deviation) * np.pi/180
        
    return dset_p_
    
def _convert_to_rectangular_coordinates(dset_p):
    """ """
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
    
def _setup_reconstruction_parameter(pixelNumber, nsamples, srate, df, modamp, demodnum, offset, demod_freqs):
    """creates and returns parameter required for the energy spectrum reconstruction 
    
    This function calculates the parameters needed for the energy spectrum reconstruction. input parameter are
    the information form the mla data header.
    
    Example:
    --------
    >>> lines = _load_mla_data(mla_data_fn)
    >>> block_list = _create_block_list(lines)
    >>> prm = _parse_mla_data_header(block_list)
    >>> t, v_t, first_index, last_index, step_size = _setup_reconstruction_parameter(**prm)

    """
    
    t = np.arange(nsamples) / srate * demod_freqs[0]
    #offset = -0.1
    v_t = offset + modamp * np.cos(2 * np.pi * t)
    #v_t = np.roll(v_t, -500)
    first_index = demod_freqs[0] / df
    last_index = demod_freqs[30] / df
    step_size = (demod_freqs[1] - demod_freqs[0]) / df
    return t, v_t, first_index, last_index, step_size

def reconstruct_energy_spectra(
        dset, prm, t, v_t, first_index, last_index, step_size
    ):
    """ """
#    arr = np.zeros([prm['pixelNumber'], prm['nsamples']], dtype='float64')
#    linarr = np.empty([prm['pixelNumber'], 313], dtype='float64')
    arr = np.zeros([dset.shape[0], prm['nsamples']], dtype='float64')
    linarr = np.empty([dset.shape[0], 313], dtype='float64')

    linearizedEnergy = np.linspace(-0.117, 0.278, 313)

    for idx, row in enumerate(dset):
        dataset = np.zeros(prm['nsamples'], dtype=np.complex)
        dataset[int(first_index): int(last_index) + int(step_size): int(step_size)] = row[0:31:1]
        recon_pixels = np.fft.ifft(dataset)
        recon_pixels_amp = 2 * np.real(recon_pixels) * 1e3
        arr[idx, :] = recon_pixels_amp
        f = interpolate.interp1d(v_t[313:625], recon_pixels_amp[313:625])
        linarr[idx, :] = f(linearizedEnergy)

#    linarr = np.diff(linarr) * -1
    linarr = np.diff(linarr)
    return linearizedEnergy, linarr, arr
    

def get_measurement_data(block_list, prm):
    """returns the phase- and amplitude corrected measurement data
    
    This function parses the complex frequency values from the text file 
    data blocks and phase- and amplitude corrects them. Returns a 2d np.arrray 
    
    Example:
    --------
    >>> lines = _load_mla_data(mla_data_fn)
    >>> block_list = _create_block_list(lines)
    >>> prm = _parse_mla_data_header(block_list)
    >>> dset = get_measurement_data(block_list, prm)
    
    
    """
    dset = _parse_all_data_blocks(block_list, prm['demodnum'])
    dset_p = _convert_to_polar_coordinates(dset)
    dset_p = _add_amplitude_and_phase_correction(dset_p)
    dset_rect = _convert_to_rectangular_coordinates(dset_p)
    # setattr(self, 'dset', dset_rect)
    return dset_rect
    
def get_energy_spectra(dset, prm):
    """ """
    t, v_t, first_index, last_index, step_size = _setup_reconstruction_parameter(**prm)

    linearizedEnergy, linarr, arr = reconstruct_energy_spectra(
        dset, prm,
        t, v_t, first_index, last_index, step_size,
    )
    return linearizedEnergy, linarr, arr
    
    
    
def generate_energy_spectra(mla_data_fn, n=1000, verbose=True):
    """loades, processes, and saves MLA energyspectra blockwise to text file.
    
    Function loads fourier coefficient from MLA measurement file, calculates
    the energy spectra, and saves the energy values to a text file. This
    process is done in a block-wise fashion to avoid loading all the data into
    memory. This function can be used to process large datasets.
    
    
    Example:
    --------
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
        linE, linarr, arr = get_energy_spectra(dset, prm)
        
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
# MLA dataset
# ===========================================================================

def _check_resize_curr(resize_curr, prm):
    """raises a ValueError if the given resize_curr doesn't match the dataset
    size specified in prm
    
    """
    err_msg = """ Given resize_curr: {0:} is not compatible with the loaded data:
        pixelNumber: {1:d}; nsamples: {2:d}""".format(
            resize_curr,
            prm['pixelNumber'],
            prm['nsamples']
    )
    
    if not prm['pixelNumber'] == np.product(resize_curr[:-1]):
        raise ValueError(err_msg)
        
    if not prm['nsamples'] == resize_curr[-1]:
        raise ValueError(err_msg)
        
    return


def _check_resize_cond(resize_cond, prm):
    
    nsamples = int(prm['nsamples']/2)
    
    err_msg = """ Given resize_cond: {0:} is not compatible with the loaded data:
        pixelNumber: {1:d}; nsamples: {2:d}""".format(
            resize_cond,
            prm['pixelNumber'],
            nsamples
    )
    
    if not prm['pixelNumber'] == np.product(resize_cond[:-1]):
        raise ValueError(err_msg)
        
    if not nsamples == resize_cond[-1]:
        raise ValueError(err_msg)
        
    return



def _load_mla_data_into_hdf5(mla_data_fn, resize_curr=False, resize_cond=False, verbose=False):
    """ 
    
    
    Example:
    --------
    >>> mla_txt_fn = '2021-03-15_TSP-MLA-zsweep_5x5nm_25x25pxl/Measurement of 2021-03-15 2005.txt'
    >>> mla_fn = _load_mla_data_into_hdf5(
    ...     mla_txt_fn,
    ...     resize_curr = (40,25,25,625),
    ...     resize_cond = (40,25,25,322)
    ... )
    
    
    
    
    """
    
    # ========================
    # load measurement data from txt file
    # ========================    
    if verbose: print('load mla data from text file')
    lines = _load_mla_data(mla_data_fn)
    block_list = _create_block_list(lines)
    prm = _parse_mla_data_header(block_list)
    data_block_list = _get_data_block_list(block_list)    

    if verbose: print('\t ...completed.\n')

    # ========================
    # check if specified arr size matches data
    # ========================        
    if resize_curr != False:
        _check_resize_curr(resize_curr, prm)
    else: 
        resize_curr = (prm['pixelNumber'], prm['nsamples'])

    if resize_cond != False:
        _check_resize_cond(resize_cond, prm)
    else: 
        resize_cond = (prm['pixelNumber'], int(prm['nsamples']/2))
    
    
    # ========================
    # create hdf5 data structure
    # ========================
    mla_hdf5_fn = mla_data_fn.replace('.txt', '.hdf5')
    f = h5py.File(mla_hdf5_fn, 'a')
    
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
    # load fft coefficients into hdf5
    # ========================
    n = 1000
    data_block_parcels = [
        data_block_list[i:i+n] for i in range(0, len(data_block_list), n)
    ]
    n_parcels = len(data_block_parcels)


    for idx, sub_block in enumerate(data_block_parcels):
        arr = get_measurement_data(sub_block, prm)
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
    # dset = _parse_all_data_blocks(block_list, prm['demodnum'])
    # dset_p = _convert_to_polar_coordinates(dset)
    # dset_p = _add_amplitude_and_phase_correction(dset_p)
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
