import SimpleITK as sitk
from pathlib import Path
from damply import dirs
import numpy as np
from tqdm import tqdm
import logging 

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("find_empty_label_image.log")
)

logger = logging.getLogger(__name__)


dataset = "CVPR_LesionLocator"

label_images_dir = dirs.RAWDATA / dataset / "images" / "Synthetic_Follow_Up" / "labels"

for file in tqdm(sorted(label_images_dir.iterdir())):
    mask = sitk.ReadImage(file)
    label_array = sitk.GetArrayFromImage(mask)
    unique_labels = np.unique(label_array)

    logger.info(f"{file.stem} labels: {unique_labels}")

    if len(unique_labels) < 2:
        logger.info(f"Label file {file.stem} has only one label: {unique_labels}")

    if not np.any(unique_labels == 1):
        logger.info(f"Label file {file.stem} has no label 1: {unique_labels}")

    
logger.info("Made it to the end.")