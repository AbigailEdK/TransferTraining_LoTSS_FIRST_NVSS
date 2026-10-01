==================================================
Process for running files in this folder, in order
==================================================

------------------------------------------------
1. combineFITStoRGB.py

Files needed: RADCAT_F folder of FITS files.

File function: Runs through all FITS files in the sub folders and combines them into a single RGB image. Red = FIRST, Green = LoTSS, Blue = NVSS. 

Final output: RADCAT_F_UNPACKED folder, where all surveys have been combined for each source. 

------------------------------------------------
2. createAugmentationsFile.py

Files needed: source_ids_test.txt, source_ids_val_train.xt, RADCAT_F_UNPACKED, RADCAT.csv.

File function: Calculates class distributions within the trainval data, after separating out test data, in the RADCAT_F_UNPACKED folder. Generates random augmentations until each class has 3000 images associated with it. 

Final output: augmentations.txt, a list of augmentations to perform on the RADCAT_F_UNPACKED images. 

------------------------------------------------
3. augmentSources.py

Files needed: augmentations.txt, RADCAT_F_UNPACKED

File function: Applies the augmentations from augmentations.txt to the images in RADCAT_F_UNPACKED and outputs them to AUGMENTED_SOURCES.

Final output: Augmented images in the AUGMENTED_SOURCES folder. 

------------------------------------------------
4. combineAllData.py

Files needed: RADCAT_F_UNPACKED, AUGMENTED_SOURCES

File function: Combines original and augmented sources into one directory. 

Final output: AGGREGATED_RGB_IMAGES folder.

------------------------------------------------
5. trainValTestSplit.py

Files needed: AGGREGATED_RGB_IMAGES, source_ids_val_train.txt, source_ids_test.txt

File function: Splits the RGB images into two folders, one containing test sources and one containing training and validation sources.  

Final output: TEST_SOURCES_RGB and TRAINVAL_SOURCES_RGB fodlers.

------------------------------------------------
6. RGBtoNumpy.py

Files needed: TRAINVAL_SOURCES_RGB and TEST_SOURCES_RGB folders

File function: To convert the RGB sources in the given folders to three numpy arrays: X, y, and associated source IDs (name). This is to be accepted by the get_dataset() function. 

Final output: Six numpy array files: X_trainval.npy, y_trainval.npy, names_trainval.npy, X_test.npy, y_test.npy, names_test.npy. 

------------------------------------------------

===================================
Auxilliary scripts and descriptions 
===================================

------------------------------------------------
>  finalDistributions.py

Files needed: AGGREGATED_RGB_IMAGES, RADCAT.csv

File function: To tally up the final distribution of classes in the aggregate files folder.

Final output: A terminal print of the final class distributions.

------------------------------------------------
> viewFITS.ipynb

Files needed: RADCAT_F

File function: To visually display all survey FITS files for a given source.

Final output: Interactive UI.

------------------------------------------------
> viewFITSandRGB.ipynb

Files needed: RADCAT_F, RADCAT_F_UNPACKED

File function: To display the component FITS files and the complete RGB image side by side. 

Final output: Interactive UI.

------------------------------------------------
> viewAugmentations.ipynb

Files needed: AUGMENTED_SOURCES, RADCAT_F_UNPACKED

File function: To display the original RGB image for a source beside all of its augmentations. 

Final output: Interactive UI.

------------------------------------------------
> trainValTestSplit.py

Files needed: RADCAT_F_UNPACKED, AGGREGATED_RGB_IMAGES

File function: To split the final RGB images, along with their augmentations, into test and trainval data. 

Final output: TRAINVAL_SOURCES_RGB, TEST_SOURCES_RGB.

------------------------------------------------
> trainValTestChannelSplit.py

Files needed: SPLIT_SOURCES, source_ids_test, source_ids_val_train

File function: To split the R, G, and B files into separate test and trainval folders. Saved as numpy files. 

Final output: TRAINVAL_SOURCES_FIRST, TRAINVAL_SOURCES_LoTSS, TRAINVAL_SOURCES_NVSS, TEST_SOURCES_FIRST, TEST_SOURCES_LoTSS, TEST_SOURCES_NVSS

------------------------------------------------

===================================
Auxilliary files and descriptions 
===================================

------------------------------------------------
>  augmentations.txt

File function: A binary text file with instructions on how to augment the RGB images in RADCAT_F_UNPACKED to synthetically augment class distribution. 
------------------------------------------------
>  source_ids_val_train.txt

File function: Source IDs for training and validation sources, according to the RADCAT.csv file and an 80/20 split. 
------------------------------------------------
>  source_ids_test.txt

File function: Source IDs for testing sources, according to the RADCAT.csv file and an 80/20 split. 
------------------------------------------------
>  augmentations_metadata.json

File function: Metadata file to track source IDs through the augmentation process. 

------------------------------------------------

============================
Final description of folders
============================

RADCAT_F: Downloaded from  https://zenodo.org/records/14718007, by Dylan Farge. A folder containing three other folders: FIRST, LoTSS, NVSS, and an augmentations text file. Each folder contains FITS files that align with the sources dictated in RADCAT.csv. 
   |
   |--- FIRST
   |--- LoTTS
   |--- NVSS
   |--- augmentations.txt

RADCAT_F_UNPACKED: All sources from the individual folders combined into one directory, as RGB images (R = FIRST, G = LoTSS, B = NVSS). 

AUGMENTED_SOURCES: All images files from RADCAT_F_UNPACKED, augmented to improve class distribution. They have been augmented according to augmentations.txt, generated by createAugmentationFile.py.

SPLIT_SOURCES: All RGB images from AGGREGATED_RGB_IMAGES split into their RGB channels - R = FIRST, G = LoTSS, B = NVSS. Saved as numpy arrays.
   |
   |--- Red (FIRST)
   |--- Green (LoTTS)
   |--- Blue (NVSS)

AGGREGATED_RGB_FILES: All RGB files from AUGMENTED_SOURCES and RADCAT_F_UNPACKED combined into one directory. 

TRAINVAL_SOURCES_RGB: All sources separated out from AGGREGATED_RGB_FILES for training and validation, according to the IDs in source_ids_val_train.txt (downloaded from https://zenodo.org/records/14718007). Images are saved as RGB images.

TRAINVAL_SOURCES_FIRST: All sources in TRAINVAL_SOURCES_RGB, stripped of B and G values to leave only Red (FIRST survey) behind. Saved as FITS files. 

TRAINVAL_SOURCES_LoTSS: All sources in TRAINVAL_SOURCES_RGB, stripped of R and G values to leave only Blue (LoTSS survey) behind. Saved as FITS files. 

TRAINVAL_SOURCES_NVSS: All sources in TRAINVAL_SOURCES_RGB, stripped of R and B values to leave only Green (NVSS survey) behind. Saved as FITS files. 

TEST_SOURCES_RGB: All sources separated out from RADCAT_F_UNPACKED for testing, according to the IDs in source_ids_test.txt (downloaded from https://zenodo.org/records/14718007). Images are saved as RGB images. 

TEST_SOURCES_FIRST: All sources separated out from RADCAT_F for testing, according to the IDs in source_ids_test.txt (downloaded from https://zenodo.org/records/14718007), for the FIRST survey.

TEST_SOURCES_LoTSS: All sources separated out from RADCAT_F for testing, according to the IDs in source_ids_test.txt (downloaded from https://zenodo.org/records/14718007), for the LoTSS survey.

TEST_SOURCES_NVSS: All sources separated out from RADCAT_F for testing, according to the IDs in source_ids_test.txt (downloaded from https://zenodo.org/records/14718007), for the NVSS survey.