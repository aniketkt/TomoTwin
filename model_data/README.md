Author: Aniket Tekawade

## Getting Beam Data from XOP  

Bending Magnet Source at APS (e.g. 7-BM)  

1. Open XOP and click Source -> Bending Magnet -> Bending Magnet Radiation  

2. In the dialog box go to File -> bm input parameters -> Load from File  

3. In XOP_setup look for bm.xop  

4. Go to Show -> 2D Plot Power (angular, energy) distribution  

5. Click Export and choose File XYZ  

6. Save this file as a .dat file, then open it in excel  

7. make columns such as below and save this file in "source_files/7BM/beam_power_7BM.csv"  

---

## Getting Materials Data from XOP  


1. Open XOP and click Optics -> Mirrors and Filters -> xCrossSec (Cross section / absorption)  
2. Click Set Parameters and in the dialog box select material (mixture formula or table)  

3. Select user-defined energy points: 1000 - 200,000 eV (996 points preferred)  

4. Select Units: cm^2/g (Mass abs coef)  

5. Click Accept to view the plot  

6. Then choose File -> Export ASCII or Excel file  

7. Type output format exactly as: XX_properties_xCrossSec.dat where "XX" is the string you would use to reference the material name in the Python API  

8. Choose "All Columns" under "Data to save" drop-down list, then click Accept  

9. It should save in a tmp folder, then move it to the model_data/materials  


    
    
