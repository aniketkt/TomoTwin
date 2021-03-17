
[![DOCS](https://readthedocs.org/projects/docs/badge/?version=latest)](https://tomotwin.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aniketkt/TomoTwin/master?filepath=notebooks%2Fsynthesize_voids_and_inclusions.ipynb)

# TomoTwin

<p align="justify">TomoTwin is a simple digital twin for synchrotron tomography that provides an API to generate realistic grayscale tomography datasets from ground-truth (labeled) volumes as training data for denoising and segmentation models. The grayscale data mimics imaging artifacts such as noise and in-line phase contrast with a dependence on the instrument parameters such as detector configuration and beam energy.</p>  

<p align="center">
  <img width="800" src="docs/source/img/schematic.png">
</p>  


## Installation  
To install using pip do this in your python environment:

```  
pip install git+https://github.com/aniketkt/TomoTwin.git@master#egg=tomo_twin  
```  

## Description  

tomo_twin  
├── gt_generators.py  
├── pg_filter.py  
├── ray.py  
└── recon.py    

### ray  
<p align="justify">Two classes are implemented: The 'Phantom' class is instantiated by passing a ground-truth labeled volume and the material name and density corresponding to each label. For example, in a porous volume mimicing rock, the voxels belonging to the matrix could be labeled '1' and those belonging to the pores could be labeled '0'. The material names and corresponding density (g/cc) would be passed as a dictionary {"air" : 0.00122, "silica" : 2.7} in that order. The 'Material' class is instantiated for each material by reading the attenuation data from model_data/materials folder. You may use XOP to generate new materials by following the README file in model_data/. Further details are in our paper.</p>  

### gt_generators  
<p align="justify">You may pass your own labeled volumes to the Phantom class. This file provides functions to parametrically create some labeled volumes that mimic porous materials, inclusions, etc. You could create a volume of up to 256 labels as this will be handled as 8-bit data.</p>  

### jupyter notebooks  
For tutorials on how to use TomoTwin, please check out the jupyter notebooks that are provided with the installation.  


## Tell me more  
Our paper is under review for IEEE-ICIP 2021 and will be up on arXiv very soon!  

<p align="center">atekawade [at] anl [dot] gov</p>  
