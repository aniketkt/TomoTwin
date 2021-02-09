#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions.

https://porespy.readthedocs.io/en/master/getting_started.html#generating-an-image  

"""
import h5py
import numpy as np
from porespy import generators
from tomopy import recon, normalize, minus_log, circ_mask
    

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
        



















    
def recon_wrapper(projs, beam, theta, pad_frac = 0.8, mask_ratio = 0.95, contrast_s = 0.001):
    
    '''
    Do a reconstruction with gridrec.  
    
    Parameters
    ----------
    projs : np.array  
        a stack of radiographs (sinogram order input to tomopy recon is False)  
        
    theta : tuple  
        The tuple must be defined as (starting_theta, ending_theta, number_projections). The angle is intepreted as degrees.  
    
    beam : np.array  
        The flat-field (beam array) must be provided with shape (1, nrows, ncols).  
        
    pad_frac : float  
        Fraction of padding applied to rows (e.g. pad_frac = 0.8 on 1000 rows adds 800 pixels to either side.  
        
    mask_ratio : float  
        ratio between (0,1) for applying circular mask on reconstruction (typical value 0.9 to 1.0).  
        
    '''
    
    # make theta array in radians
    theta = np.linspace(*theta, endpoint = True)
    theta = np.radians(theta)
    
    projs = normalize(projs, beam, 1.0e-6*np.zeros(beam.shape))
    projs = minus_log(projs)
    
    pad_w = int(pad_frac*projs.shape[-1])
    projs = np.pad(projs, ((0,0), (0,0), (pad_w, pad_w)), mode = "constant", constant_values = 0.0)
    rec = recon(projs, theta = theta, \
                  center = projs.shape[-1]//2, \
                  algorithm = 'gridrec', \
                  sinogram_order = False)
    rec = rec[:, pad_w:-pad_w, pad_w:-pad_w]
    rec = circ_mask(rec, 0, ratio = mask_ratio)
    mask_val = rec[int(rec.shape[0]//2), 0, 0]
    vcrop = int(rec.shape[0]*(1-mask_ratio))
    rec[0:vcrop,...] = mask_val
    rec[-vcrop:,...] = mask_val
    
    if contrast_s > 0.0:
        h = modified_autocontrast(rec, s = contrast_s)
        rec = np.clip(rec, *h)
    
    return rec    
    
    
def calc_SNR(img, seg_img, labels = (0,1), mask_ratio = None):
    """
    SNR =  1     /  s*sqrt(std0^^2 + std1^^2)  
    where s = 1 / (mu1 - mu0)  
    mu1, std1 and mu0, std0 are the mean / std values for each of the segmented regions respectively (pix value = 1) and (pix value = 0).  
    seg_img is used as mask to determine stats in each region.  

    Parameters
    ----------
    img : np.array  
        raw input image (2D or 3D)  
    
    seg_img : np.array  
        segmentation map (2D or 3D)  
        
    labels : tuple  
        an ordered list of two label values in the image. The high value is interpreted as the signal and low value is the background.  
        
    mask_ratio : float or None
        If not None, a float in (0,1). The data are cropped such that the voxels / pixels outside the circular mask are ignored.  

    Returns
    -------
    float
        SNR of img w.r.t seg_img  

    """
    
    # handle circular mask  
    if mask_ratio is not None:
        crop_val = int(img.shape[-1]*0.5*(1 - mask_ratio/np.sqrt(2)))
        crop_slice = slice(crop_val, -crop_val)    

        if img.ndim == 2: # 2D image
            img = img[crop_slice, crop_slice]
            seg_img = seg_img[crop_slice, crop_slice]
        elif img.ndim == 3: # 3D image
            vcrop = int(img.shape[0]*(1-mask_ratio))
            vcrop_slice = slice(vcrop, -vcrop)
            img = img[vcrop_slice, crop_slice, crop_slice]
            seg_img = seg_img[vcrop_slice, crop_slice, crop_slice]
            
            

        
    pix_1 = img[seg_img == labels[1]]
    pix_0 = img[seg_img == labels[0]]
    mu1 = np.mean(pix_1)
    mu0 = np.mean(pix_0)
    s = abs(1/(mu1 - mu0))
    std1 = np.std(pix_1)
    std0 = np.std(pix_0)
    std = np.sqrt(0.5*(std1**2 + std0**2))
    std = s*std
    return 1/std
    
    
    
    
def get_cropslices(vol_shape, mask_ratio):
    
    '''
    
    Returns
    -------
    tuple
        crop_slice, vcrop_slice
        
    '''
    if mask_ratio is not None:
        crop_val = int(vol_shape[-1]*0.5*(1 - mask_ratio/np.sqrt(2)))
        crop_slice = slice(crop_val, -crop_val)    
        vcrop = int(vol_shape[0]*(1-mask_ratio))
        vcrop_slice = slice(vcrop, -vcrop)
    else:
        crop_slice, vcrop_slice = None, None
    return crop_slice, vcrop_slice    

    
def modified_autocontrast(vol, s = 0.01):
    
    '''
    Returns
    -------
    tuple
        alow, ahigh values to clamp data  
    
    Parameters
    ----------
    s : float
        quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels
    
    '''
    
    data_type  = np.asarray(vol).dtype
    
    
    if type(s) == tuple and len(s) == 2:
        slow, shigh = s
    else:
        slow = s
        shigh = s

    h, bins = np.histogram(vol, bins = 500)
    c = np.cumsum(h)
    c_norm = c/np.max(c)
    
    ibin_low = np.argmin(np.abs(c_norm - slow))
    ibin_high = np.argmin(np.abs(c_norm - 1 + shigh))
    
    alow = bins[ibin_low]
    ahigh = bins[ibin_high]
    
    return alow, ahigh
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    