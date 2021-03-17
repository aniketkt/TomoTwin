#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  
Wrapper functions for tomopy reconstruction.  

"""
import numpy as np
from tomopy import recon, normalize, minus_log, circ_mask
    

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
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
