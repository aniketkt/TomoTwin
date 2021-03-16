#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:59:34 2019

@author: atekawade
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import misc
import h5py
import time
import configargparse
import ast

import matplotlib as mpl
import matplotlib.pyplot as plt
from ct_segnet import viewer
mpl.use('Agg')
figw = 12

if "../tomo_twin" not in sys.path: # local imports
    sys.path.append('../tomo_twin')
import ray_utils as ray
from utils import *


def custom_tuple(s):
    """
    tuple with custom split string.
    """
    s = s.split('x')
    s = ','.join(s)
    return ast.literal_eval(s)

def custom_dict(s):
    """
    dictionary with split string.
    """
    s = s.split(':')
    s = ':'.join(s)
    return ast.literal_eval(s)
    


def main(args):
    print("\n" + "#"*70)
    print("\n\tTomoTwin: Simple Digital Twin for Synchrotron micro-CT")
    print("\n\tauthor: Aniket Tekawade; atekawade [at] anl [dot] gov")
    print("\n" + "#"*70 + "\n")
    
    
    if os.path.exists(args.save_path):
        if not args.overwrite:
            raise ValueError("File already exists and overwrite disabled")
            
    fname = os.path.split(args.save_path)[-1].split('.hdf5')[0]
    
    
    plots_path = os.path.join(args.save_plot_path, fname)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    
    proj_shape = args.FOV
    args.energy_pts = np.asarray(args.energy_pts)
    print("Shape of the projection image: %s"%str(proj_shape))
    print("FOV of %.1f mm high, %.1f mm wide; pixel size %.2f um"%(args.FOV[0]*args.res/1000, args.FOV[1]*args.res/1000, args.res))
    
    mat_dict = dict(zip(args.materials, args.densities))
    
    obj_shape = (proj_shape[0], proj_shape[1], proj_shape[1])
    
    

    beam = ray.read_source(args.source_file_path, \
                           args.energy_pts, \
                           res = args.res, \
                           img_shape = proj_shape)
    print("beam array shape is n_energies, nz, nx: %s"%str(beam.shape))
    

    
    # make_voids_inclusions is defined in the utils.py file  
    vol = make_voids_inclusions(obj_shape, void_frac = args.void_frac, \
                                void_size = args.void_size, \
                                inclusion_frac = args.inclusion_frac, \
                                inclusion_size = args.inclusion_size)    
    
    Ph = ray.Phantom(vol, mat_dict, args.res, args.energy_pts)
    
    fig, ax = plt.subplots(1,2, figsize = (figw,figw/2))
    h = ax[0].imshow(vol[vol.shape[0]//2], cmap = 'gray')
    h = ax[0].set_title("XY mid-plane", fontsize = 14)
    h = ax[1].hist(vol.reshape(-1), bins = 3)
    h = ax[1].set_ylabel("voxel count", fontsize = 14)
    h = ax[1].set_xlabel("void, material, inclusions", fontsize = 14)
    plt.savefig(os.path.join(plots_path, fname + "_phantom_hist.png"))
    
    fig, ax = plt.subplots(1,3, figsize = (figw,figw/3))
    h = viewer.view_midplanes(vol = Ph.vol, ax = ax)
    plt.savefig(os.path.join(plots_path, fname + "_phantom_midplanes.png"))
    
    # Make projections
    t0 = time.time()
    theta = (0, 180, args.n_projections)
    projs = Ph.get_projections(theta = theta, beam = beam, \
                               noise = args.noise, \
                               detector_dist = args.detector_dist, \
                               blur_size = args.blur_size)
    tot_time = (time.time() - t0)/60.0
    print("Shape of the synthesized projections data: %s"%str(projs.shape))
    print("Done in %.2f minutes"%tot_time)    
    
    
    fig, ax = plt.subplots(1,2, figsize = (figw,figw/2))
    h = ax[0].imshow(projs[0], cmap = 'gray')
    h = ax[0].set_title("Projection at %.2f degrees"%theta[0], fontsize = 14)
    h = ax[1].hist(projs[0].reshape(-1), bins = 200)
    fig.tight_layout()    
    plt.savefig(os.path.join(plots_path, fname + "_projections_hist.png"))
    
    
    select_rows = [0.1,0.5,0.9]
    select_idx = [int(select_row*projs.shape[1]) for select_row in select_rows]
    fig, ax = plt.subplots(1,len(select_rows), figsize = (figw,figw/len(select_rows)))
    for ii, idx in enumerate(select_idx):
        ax[ii].imshow(projs[:,idx,:], cmap = 'gray')
        ax[ii].set_title("Row at %.1f pc from top"%(100.0*select_rows[ii]), fontsize = 14)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_path, fname + "_sinograms.png"))
    
    
    
    # Do a test reconstruction  
    rec = recon_wrapper(projs, beam, theta, pad_frac = args.pad_frac, mask_ratio = args.mask_ratio, contrast_s = args.contrast_s)
    
    fig, ax = plt.subplots(1,3, figsize = (figw, figw/3))
    h = viewer.view_midplanes(rec, ax = ax)
    plt.savefig(os.path.join(plots_path, fname + "_test_recon.png"))
    
    SNR_voids = calc_SNR(rec, vol, labels = (0,1), mask_ratio = args.mask_ratio)
    SNR_inclusions = calc_SNR(rec, vol, labels = (1,2), mask_ratio = args.mask_ratio)
    print("SNR of material against voids: %.2f"%SNR_voids)
    print("SNR of inclusions in material: %.2f"%SNR_inclusions)
    
    fig, ax = plt.subplots(1,2, figsize = (figw, figw/2))
    idx_mid = int(rec.shape[0]//2)
    viewer.edge_plot(rec[idx_mid], vol[idx_mid] == 0, ax = ax[0], color = [255,0,0])
    ax[0].set_title("SNR = %.2f"%calc_SNR(rec[idx_mid], vol[idx_mid], labels = (0,1), mask_ratio = args.mask_ratio))
    ax[1].set_title("SNR = %.2f"%calc_SNR(rec[idx_mid], vol[idx_mid], labels = (1,2), mask_ratio = args.mask_ratio))
    viewer.edge_plot(rec[idx_mid], vol[idx_mid] == 2, ax = ax[1])    
    plt.savefig(os.path.join(plots_path, fname + "_test_recon_labels.png"))
    
    
    hf = h5py.File(args.save_path, 'w')
    hf.create_dataset("gt_labels", data = vol)
    hf.create_dataset("projs", data = projs)
    hf.create_dataset("test_rec", data = rec)
    
    return



if __name__ == "__main__":
    
    
    parser = configargparse.ArgParser()
    
    parser.add('-c', '--config-setupseg', required=True, is_config_file=True, help='config file for segmenter')
    parser.add('-f', '--save_path', required = True, type = str, help = 'path to save synthetic data to')
    
    parser.add('--overwrite', required = False, default = False, type = bool, help = 'overwrite output file if already exists')
    parser.add('--source_file_path', required = True, type = str, help = 'path to source (beam power) data file')
    parser.add('--save_plot_path', required = True, type = str, help = 'path to save plots')
    
    
    # Important parameters to set FOV and beam (flat-field)  
    parser.add('--res', required = True, type = float, help = 'pixel size in microns')
    parser.add('--FOV', required = True, type = custom_tuple, help = 'FOV as HxV in pixels')
    parser.add('--energy_pts', required = True, type = float, action = 'append', help = 'energy or list of energies in keV')
    
    
    # Parameters for synthetic projection data  
    parser.add('--n_projections', required = True, type = int, help = 'number of projections')
    parser.add('--noise', required = True, type = float, help = 'the noise std. dev. based on the Poisson model is increased by that factor.')
    parser.add('--detector_dist', required = False, type = float, default = 0.0, help = 'distance between sample and detector (in-line phase contrast)')
    parser.add('--blur_size', required = False, type = int, default = 3, help = 'kernel size for applying blur on projection images (emulates detector blur)')
    
    # Parameters for the GT phantom
    parser.add('--materials', required = True, type = str, action = 'append', help = 'list of material names')
    parser.add('--densities', required = True, type = float, action = 'append', help = 'list of respective densities of materials in g/cc}')
    parser.add('--void_frac', required = True, type = float, help = 'number in range (0,1) - ideally < 0.5')
    parser.add('--void_size', required = True, type = float, help = 'number in (0,2), lower value results in numerous, tinier voids')
    parser.add('--inclusion_frac', required = True, type = float, help = 'number in (0,1), fraction of volume occupied by inclusions')
    parser.add('--inclusion_size', required = True, type = float, help = 'number in (0,2), lower value results in numerous, tinier inclusions')  

    # Parameters for reconstruction test
    parser.add('--mask_ratio', required = True, type = float, help = 'ratio of mask to be applied on reconstructed volume')  
    parser.add('--contrast_s', required = True, type = float, help = 'contrast adjustment')  
    parser.add('--pad_frac', required = True, type = float, help = 'padding fraction for reconstruction')  
    
    
    args = parser.parse_args()
    
    time_script_start = time.time()
    main(args)
    time_script_end = time.time()
    tot_time_script = (time_script_end - time_script_start)/60.0
    
    _str = "Total time elapsed for script: %.2f minutes"%tot_time_script 
    
    
    
    
               
    
