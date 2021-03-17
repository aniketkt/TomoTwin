#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  
Functions to generate ground-truth phantoms. We rely heavily on porespy.  
https://porespy.readthedocs.io/en/master/getting_started.html#generating-an-image  

"""
import h5py
import numpy as np
from porespy import generators
    

def make_voids_inclusions(obj_shape, param_dict = None, \
                          void_frac = 0.2, \
                          void_size = 1.25, \
                          inclusion_frac = 0.05, \
                          inclusion_size = 0.15):
    '''
    Makes a phantom of an Aluminum alloy (label = 1) with big seeded voids (label = 0) and inclusions (label = 2)   
    
    Parameters
    ----------
    obj_shape : tuple  
        shape nz, ny, nx  
    
    param_dict : dict or None
        dictionary of parameters void_frac, void_size, inclusion_frac, inclusion_size or provide them separately.  
        
    void_frac : float  
        number in range (0,1) - ideally < 0.5  
    
    void_size : float  
        number in (0,2), lower value results in numerous, tinier voids  
        
    inclusion_frac : float  
        number in (0,1), fraction of volume occupied by inclusions    
        
    inclusion_size : float  
        number in (0,2), lower value results in numerous, tinier inclusions  
        
    '''
    
    if param_dict is not None:
        void_frac = param_dict["void_frac"]
        void_size = param_dict["void_size"]
        inclusion_frac = param_dict["inclusion_frac"]
        inclusion_size = param_dict["inclusion_size"]
        
    vol_f = generators.blobs(shape=obj_shape, \
                           porosity=1-void_frac, \
                           blobiness=10.0**(1-void_size)).astype(np.uint8)
    
    
    if inclusion_frac > 0.0:
        vol = generators.blobs(shape=obj_shape, \
                               porosity=inclusion_frac, \
                               blobiness=10.0**(1-inclusion_size)).astype(np.uint8)
        vol_f[vol_f == 1] = vol[vol_f == 1] + vol_f[vol_f == 1]
    
    return vol_f    


def make_porous_material(obj_shape, void_frac = 0.2, void_size = 1.25):
    '''
    Makes a phantom of a porous material with voids of various sizes (label = 1 is material, label = 0 is void)     
    
    Parameters
    ----------
    obj_shape : tuple  
        shape nz, ny, nx  
    
    void_frac : float  
        number in range (0,1) - ideally < 0.5  
    
    void_size : float  
        number in (0,2), lower value results in numerous, tinier voids  
        
    '''
    
    if type(void_frac) is not list:
        void_frac = [void_frac]
        void_size = [void_size]
    
    vol = np.ones(obj_shape, dtype = np.uint8)
    
    for idx, vf in enumerate(void_frac):
        vol = vol*generators.blobs(shape=obj_shape, \
                                   porosity=1-vf, \
                                   blobiness=10.0**(1-void_size[idx])).astype(np.uint8)
        
    return vol


def add_water(vol, water_frac = 0.2, blob_size = 1.25):

    '''
    Adds a "water" phase to the voids in a porous matrix  
    
    Returns
    -------
    
    np.array
        label = 0 is void; label = 1 is water, label = 2 is rock (material)
    
    Parameters
    ----------
    vol : np.array  
        porous phantom with shape nz, ny, nx. label = 1 is material, label = 0 is void.  
    
    water_frac : float  
        number in range (0,1) - ideally < 0.5  
    
    blob_size : float  
        number in (0,2), lower value results in numerous, tinier water regions  
        
    '''
    
    vol[vol == 1] = 3 # label 2 is now material
    
    vol = vol + generators.blobs(shape = vol.shape, \
                             porosity = water_frac, \
                             blobiness = 10.0**(1-blob_size))
    vol[vol > 1] = 2
    
    return vol

    
def make_inclusions(obj_shape, inclusion_frac = 0.2, inclusion_size = 1.25):
    '''
    Makes a phantom of a material with inclusions of various sizes (label = 0 is material, label = 1 is inclusion)     
    
    Parameters
    ----------
    obj_shape : tuple  
        shape nz, ny, nx  
    
    inclusion_frac : float  
        number in range (0,1) - ideally < 0.5  
    
    inclusion_size : float  
        number in (0,2), lower value results in numerous, tinier voids  
        
    '''
    
    vol = make_porous_material(obj_shape, void_frac = inclusion_frac, \
                               void_size = inclusion_size)
    
    vol = vol^1
    return vol

from scipy import stats
def make_spheres(obj_shape, void_frac = 0.5, radius = 8, radius_std = 4, nbins = None):
    '''
    Makes a phantom of a material with spherical voids of different sizes (label = 1 is material, label = 0 is spherical void)     
    
    Parameters
    ----------
    obj_shape : tuple  
        shape nz, ny, nx  
    
    void_frac : float  
        number in range (0,1) - ideally < 0.5  
    
    radius : float
        number in voxels indicating mean radius  
        
    radius_std : float
        number in voxels indicating standard deviation of distribution of radii  
    
    nbins : int
        these many discrete values of sphere radii will be generated
        
    '''
    
    if nbins is None:
        nbins = 2*radius_std + 1
        
    dist = stats.norm(loc = radius, scale = radius_std)
    
    vol = generators.polydisperse_spheres(shape = obj_shape, \
                                          porosity = 1 - void_frac, \
                                          dist = dist, nbins = nbins)
    
    return vol





def make_fibrousmat(obj_shape, radius = 8, ncylinders = 50, radius_std = 4, theta_max = 0.1, phi_max = 0.1):
    '''
    Makes a phantom of a material with spherical voids of different sizes (label = 1 is material, label = 0 is spherical void)     
    
    Parameters
    ----------
    obj_shape : tuple  
        shape nz, ny, nx  
    
    ncylinders : int  
        number of cylinders total  
        
    radius : float
        number in voxels indicating mean radius  
        
    radius_std : float
        number in voxels indicating standard deviation of uniform distribution of radii  
        
    '''

    nbins = int(2*radius_std+1)
    radius_list = np.linspace(radius-radius_std, \
                              radius+radius_std, \
                              nbins, \
                              endpoint = True)
    
    vol = np.ones(obj_shape, dtype = np.uint8)
    
    for rad in radius_list:
        vol = vol*generators.cylinders(shape = obj_shape, \
                                       radius = rad, \
                                       ncylinders = ncylinders//nbins, \
                                       theta_max = theta_max, \
                                       phi_max = phi_max)
        
    return vol
        


