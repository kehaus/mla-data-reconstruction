#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:47:56 2021

@author: kh
"""

import os
import sys

### =========================================================================
#%% import mla_reconstruction function
### =========================================================================
repo_dir = os.path.dirname(os.getcwd())
src_dir = os.path.join(
    repo_dir,
    'src'
)
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
### =========================================================================
#%% load test data
### =========================================================================
data_dir = os.path.join('examples', 'test_data')
data_subdir1 = '2021-03-19_grid-MLA-zsweep_5x5nm_5x5pxl_test'
fn_txt1 = os.path.join(
    repo_dir, data_dir, data_subdir1, 
    'Measurement of 2021-03-20 0655.txt'
)

data_subdir2 = '20210604_MLA_DCoffset_test'
fn_txt2 = os.path.join(
    repo_dir, data_dir, data_subdir2,
    'Measurement of 2021-06-04 1734.txt'
)


# ### =========================================================================
# #%% test loead-header function
# ### =========================================================================
# from mla_reconstruction import _load_mla_data_header


# # ==== Test case 1: load from str (should work)
# f = open(fn_txt, mode='r')
# prm = _load_mla_data_header(f, pixel_number=1000)
# f.close()

# # ==== Test case 2: load from file handler (should work)
# prm = _load_mla_data_header(fn_txt, pixel_number=1000)
# f.close()


# # ==== Test case 3: load from non-existant file (should raise MLAReconstructionException)
# prm = _load_mla_data_header('wrong_filename.txt', pixel_number=1000)
# f.close()

# # ==== Test case 4: load with wrong variable type (should raise TypeError)
# prm = _load_mla_data_header(17, pixel_number=1000)



# ### =========================================================================
# #%% test parse-header function
# ### =========================================================================
# from mla_reconstruction import _load_mla_data, _create_block_list, _parse_mla_data_header


# lines = _load_mla_data(fn_txt)
# block_list = _create_block_list(lines)
# hd_prm = _parse_mla_data_header(block_list)




### =========================================================================
#%% test txt to hdf5 conversion -- Non offset
### =========================================================================
from mla_reconstruction import _load_mla_data_into_hdf5

# ==== Test case 1: load and reshape
fn_h5 = _load_mla_data_into_hdf5(
    fn_txt1,
    resize_curr=(5,5,40,119),
    resize_cond=(5,5,40,119),
    pixel_number=1000,
    zsweep_nr=40,
    mode='w',
    # missing_pxls=[10,60,135]
)




