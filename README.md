# TomoTwin: A simple digital twin for synchrotron micro-CT

Author: Aniket Tekawade; atekawade [at] anl [dot] gov


A typical example for generating synthetic data with this code:  

1. Phantom with voids and inclusions using Porespy (Label 0 is void, label 1 is material, label 2 is inclusion material)  
2. model attenuation / noise with Poisson assumption and data from XOP  
3. model phase-contrast with inverse phase-retrieval step (assuming refractive index is proportional to absorption coefficient  
4. model beam profile with XOP data on bending magnet power distribution at 35 m from source
