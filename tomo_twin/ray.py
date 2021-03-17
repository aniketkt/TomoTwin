#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Phantom class is instantiated with a ground-truth phantom and corresponding material properties data. The get_projections method simulates data acquisition and returns radiographs for the specified theta values.  

"""


import sys
import os
import numpy as np
import pandas as pd
from scipy import misc
import h5py
import time
from scipy.integrate import simps

import matplotlib.pyplot as plt

import cv2
from tomopy import project
from scipy.ndimage.filters import gaussian_filter
from tomo_twin.pg_filter import add_phase_contrast

model_data_path = '../model_data'

class Phantom:
    
    def __init__(self, vol, materials, res, energy_pts, bits = 16, data_path = model_data_path):
        '''
        Parameters
        ----------
        
        vol : np.array
            labeled (segmented / ground-truth) volume. voxel values are in finite range [0,...n_materials-1].  
            
        materials : dict
            dict of material names and their respective density g/cc, e.g. {"Fe" : 7.87, "Al": 2.7}    
            
        res : float
            voxel size in microns  
        
        energy_pts : float or np.array
            list of energies
            
        bits : int
            16 for 16 bit camera  
            
        data_path : str
            path to exported XOP data
            
        '''
        
        # deal with materials
        self.bits = bits
        self.res = res
        self.data_path = data_path
        self.energy_pts = np.asarray(energy_pts) if type(energy_pts) is float else energy_pts
        self.materials = [Material(key, value, \
                                   self.res, \
                                   self.energy_pts, \
                                   data_path = self.data_path) for key, value in materials.items()]
        self.sigma_mat = np.concatenate([material.sigma for material in self.materials], axis = 1)
        
        # some numbers
        self.n_mat = len(self.materials)
        self.n_energies = np.size(self.energy_pts)
        
        # deal with labeled volume
        self.vol = vol
        self.vol_shape = self.vol.shape
        
        if self.vol.max() != (len(self.materials)-1):
            raise ValueError("Number of materials does not match voxel value range.")
        
        if len(self.vol_shape) not in (2,3): raise ValueError("vol must have either 2 or 3 dimensions.")
        self.ray_axis = 1 if  len(self.vol_shape) == 3 else 0
        
        if len(self.vol_shape) == 3:
            self.proj_shape = (self.vol_shape[0], self.vol_shape[-1])
        else:
            self.proj_shape = (self.vol_shape[-1],)
        
        self.make_volume() # blows up volume into individual energies
        
    
    def make_volume(self):
        '''
        Converts the labeled GT volume provided into a volume of sigma values (attenutation coefficient, density and pixel size as pathlength). The resulting shape is (nz, ny, nx) or (n_energies, nz, ny, nx). The "energy" channel is added if multiple energies are requested.  
        '''
        
        voxel_vals = np.arange(self.n_mat)
        self.vol = np.asarray([self.vol]*self.n_energies, dtype = np.float32)
        
        for ie in range(self.n_energies):
            for voxel in voxel_vals:
                self.vol[ie,  self.vol[ie] == voxel] = self.sigma_mat[ie,voxel]
        
        if self.n_energies == 1:
            self.vol = self.vol[0]
            return
        else:
            return

    
    
    def get_projections(self, theta = (0,180,180), beam = None, noise = 0.01, blur_size = 5, detector_dist = 0.0):

        
        '''
        Acquire projections on the phantom.  
        Returns  
        -------  
        np.array  
            output shape is a stack of radiographs (nthetas, nrows, ncols)  
        Parameters  
        ----------  
        theta : tuple  
            The tuple must be defined as (starting_theta, ending_theta, number_projections). The angle is intepreted as degrees.  
        beam : np.array  
            The flat-field (beam array) must be provided with shape (1, nrows, ncols) or (n_energies, nrows, ncols).  
        noise : float
            The noise parameter is interpreted as a fraction (0,1). The noise transforms the pixel map I(y,x) in the projection space as I(y,x) --> I(y,x)*(1 + N(mu=0, sigma=noise)).  
        
        '''
        
        # make theta array in radians
        theta = np.linspace(*theta, endpoint = True)
        theta = np.radians(theta)
        
        # make beam array (if not passed)
        if beam is None:
            beam = np.ones(self.proj_shape, dtype = np.float32)
            beam = beam*(2**self.bits-1)
        
        # if monochromatic beam
        if self.n_energies == 1:
            
            projs = project(self.vol, theta, pad = False, emission = False)
            projs = projs*beam
            
            # scintillator / detector blurring  
            if blur_size > 0:
                projs = [proj for proj in projs]
                projs = Parallelize(projs, gaussian_filter, \
                                    procs = 12, \
                                    sigma = 0.3*(0.5*(blur_size - 1) - 1) + 0.8, \
                                    order = 0)
                projs = np.asarray(projs)
            
            # in-line phase contrast based on detector-sample distance (cm)
            if detector_dist > 0.0:
                
                pad_h = int(projs.shape[1]*0.4)
                projs = np.pad(projs, ((0,0), (pad_h,pad_h), (0,0)), mode = 'reflect')
                projs = add_phase_contrast(projs, \
                                           pixel_size = self.res*1e-04, \
                                           energy = float(self.energy_pts), \
                                           dist = detector_dist)
                projs = projs[:,pad_h:-pad_h,:]
            
            # Poisson noise model (approximated as normal distribution)
            projs = np.random.normal(projs, noise*np.sqrt(projs))
#             projs = np.random.poisson(projs)

        # This actually worked fine
#             projs = projs*beam*(1 + np.random.normal(0, noise, projs.shape))
        
        
        # if polychromatic beam
        else:
            projs = Parallelize(theta.tolist(), \
                                _project_at_theta, \
                                vol = self.vol, \
                                n_energies = self.n_energies, \
                                beam = beam, \
                                noise = noise, procs = 12)
            projs = np.asarray(projs)
        
        # saturated pixels
        projs = np.clip(projs, 0, 2**self.bits-1)
        
        return projs.astype(np.uint16)
                
    
class Material:
    
    # Ideas borrowed from Alan Kastengren's code for BeamHardeningCorrections (7-BM github)
    def __init__(self, name, density, path_len, energy_pts, scintillator_flag = False, data_path = None):
        """
        Parameters
        ----------
        name            : str
            string describing material name. Typically, use chemical formula, e.g. Fe, Cu, etc.  
            
        density         : float
            g/cm3 units  
            
        path_len        : float
            thickness for components (filters, scintillators, etc.) and pixel size for materials in phantom  
            
        energy_pts        : np array
            listing the energy_pts requested. shape is (n,)  
            
        scintillator_flag : bool
            return absorption data instead of attenuation, if material is scintillator  
        
        sigma             : np.array  
            sigma array with dimensions (n_energies, 1)  
            
        att_coeff         : np.array  
            mass attenuation coefficient array (n_energies, 1)  
            
        data_path : str
            path to exported XOP data
        
        """
        self.name = name
        self.data_path = data_path
        self.density = density # g/cc
        self.scintillator_flag = scintillator_flag
        self.path_len = path_len # um
        self.energy_pts = energy_pts
        self.calc_sigma()
        
    def read_attcoeff(self):
        """
        # att_coeff  : cm2/g units,      array dimensions of (n_energies,)
        """
        df = pd.read_csv(os.path.join(self.data_path, 'materials', self.name + "_properties_xCrossSec.dat"), sep = '\t', delimiter = " ", header = None)
    
        old_energy_pts = np.asarray(df[0])/1000.0
        if self.scintillator_flag:
            att_coeff = np.asarray(df[3])
        else:
            att_coeff = np.asarray(df[6])
        
        self.att_coeff = np.interp(self.energy_pts, old_energy_pts, att_coeff).reshape(-1,1)
    
    def calc_sigma(self):
        
        self.read_attcoeff()
        self.sigma = np.multiply(self.att_coeff, self.density)*(self.path_len*1e-4) # att_coeff in cm2/g, rho in g/cm3, res in cm

def read_source(file_path, energy_pts, res = 1.17, img_shape = (1200,1920), bits = 16, exp_fac = 0.92):
    """
    
    Reads data from a source hdf5 file, in a format specific to this code. The original data is adapted from DABAX in XOP.
    returns b : beam array shape (n_energies, V, H) or (n_energies, 1)
    
    Two choices:
    1. enter 2D shape to incorporate vertically varying fan beam profile and spectral variation. If 2D, crops the source within the FOV of Camera defined by (res, shape). Assumes FOV is in vertical center of fan beam.  
    2. enter 1D shape to ignore and get only spectral variation.  
    
    Parameters
    ----------
    file_path         : str  
        filepath for reading beam source, e.g. bending magnet, undulator or monochromatic source, etc.  
    energy_pts        : np.array
        energy points in keV, array with dimensions (n_energies,)  
    res               : float  
        pixel resolution of camera in micrometers  
    shape             : np.array  
        pixel array size V, H  
    """
    if type(energy_pts) is float:
        energy_pts = np.asarray([energy_pts])
    # Check shape to find if 2D or 1D beam is requested. If 1D, the beam is same across H, so return a 0D beam.
    if len(img_shape) == 1:
        H = img_shape
        V = 0
    else:
        V, H = img_shape
        
    # Read from hdf5 file. This is a specific format for this code. Use "make_sourcefile.py" to create your own with XOP (X,Y,Z) csv file as input.
    with h5py.File(file_path, 'r') as hf:
        b = np.asarray(hf["power"][:])
        old_energy_pts = np.asarray(hf["energy_pts"][:])/1000.0
        mm = np.asarray(hf["vert_mm"][:])
    
    if V != 0:
        # Crop out the beam which is not in field of view. E.g. V = 1200 pixels, res = 1.17 um --> size = 1.4 mm, which is +/- 0.02 mrads
        mm_low, mm_high = -res*0.001*V/2, res*0.001*V/2 # 0.001 to convert res in um to mm
        idx = np.where((mm > mm_low) & (mm < mm_high))
        b = b[idx,...][0]

        # Resize energy_pts as requested
        b = np.asarray([np.interp(energy_pts, old_energy_pts, b[ii,...]) for ii in range(b.shape[0])])

        # Create bright-field image with beam profile along Y, duplicates along X
        b = cv2.resize(b, (energy_pts.size, V))
        b = np.tile(b[:,np.newaxis,:], (1, H, 1))
        b = np.moveaxis(b, 2, 0)
    
    else:
        b = np.mean(b, axis = 0)
        b = np.interp(energy_pts, old_energy_pts, b)
        b = b.reshape(-1,1)


    eps = 1.0e-12
    b_min = np.min(b)
    b_max = np.max(b)
    b = (b - b_min) / (b_max - b_min)
    b = b/energy_pts.size
    
    b = (2**bits-1 - 2000)*(exp_fac*b + (1-exp_fac))
    
    return b 


import functools
from multiprocessing import Pool, cpu_count



def _project_at_theta(theta_val, vol = None, n_energies = None, beam = None, noise = None):

    proj = np.asarray([project(vol[ie], \
                                np.radians([theta_val]), \
                                pad = False, \
                                emission = False)[0] for ie in range(n_energies)])
    proj = proj*beam + np.random.normal(0, noise/n_energies, beam.shape)
    proj = simps(proj, x = energy_pts, axis = 0)
    return proj




def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    """
    This function packages the "starmap" function in multiprocessing, to allow multiple iterable inputs for the parallelized function.  
    
    Parameters
    ----------
    ListIn: list
        each item in the list is a tuple of non-keyworded arguments for f.  
    
    f : function
        function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.  
    """  
    
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        opt_procs = int(np.interp(len(ListIn), [1,100,500,1000,3000,5000,10000] ,[1,2,4,8,12,36,48]))
        procs = min(opt_procs, cpu_count())

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList








































    
