#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:51:59 2019

@author: atekawade
"""
import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt





##### Make file for 7BM Bending Magnet source #######
input_filename = 'beam_power_7BM.csv'
output_filename = 'beam_profile_7BM.hdf5'
n_energies = 1991
vert_coordinates = 501
source_dist = 35.0


if __name__ == "__main__":
    df = pd.read_csv(input_filename)

    df = df.sort_values(by = ['mrad', 'keV'])    
    b = np.asarray([[df.iloc[ii + n_energies*jj]['Power'] for ii in range(n_energies)] for jj in range(vert_coordinates)])

    mrad = np.sort(df['mrad'].unique())
    mm = source_dist*mrad
    energy_pts = np.sort(df['keV'].unique())
    
    with h5py.File(output_filename, 'w') as hf:
        hf.create_dataset("power", data = b)
        hf.create_dataset("energy_pts", data = energy_pts)
        hf.create_dataset("vert_mm", data = mm)



