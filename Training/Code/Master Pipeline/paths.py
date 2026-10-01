import os

HOME_DIR = os.path.expanduser("~")
HOME_DIR = os.path.join(HOME_DIR, "Desktop") 
PROJECT_ROOT = os.path.join(HOME_DIR, "MNRAS_Paper_Code", "Training")
DATA_ROOT = os.path.dirname(PROJECT_ROOT)
MANAGER = os.path.join(PROJECT_ROOT, "Code", "Master Pipeline", "masterManager.py")
RUNMODEL = os.path.join(PROJECT_ROOT, "Code", "Master Pipeline", "masterRunModel.py")
AUGMENTDATA = os.path.join(PROJECT_ROOT, "Code", "Master Pipeline", "masterDataAugmentation.py")
RESULTSDIRECTORY = os.path.join(PROJECT_ROOT, "Results")
SOURCEDATADIRECTORY = os.path.join(RESULTSDIRECTORY, "SourceData")

FITS_DIR = os.path.join(DATA_ROOT, "Data", "FITS_BY_CHANNEL")
SOURCE_DIR = os.path.join(DATA_ROOT, "Data", "DATA_SOURCES")
CATALOG_FILE = os.path.join(DATA_ROOT, "Data", "RADCAT.csv")

