import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk

from imgtools.coretypes import MedImage, Mask, VectorMask
from pathlib import Path
from skimage.measure import regionprops
from damply import dirs

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("lesion_locator_processing.log")
)

logger = logging.getLogger(__name__)


def mask2D_to_oriented_bbox(mask:np.array) -> np.array:
	"""Convert a 2D binary mask to an oriented bounding box around the region of interest"""
	props = regionprops(mask)[0]
	y_cent, x_cent = props.centroid
	orientation = props.orientation
	semi_maj_axis_len = props.axis_major_length / 2

	x_start = x_cent - np.sin(orientation) * semi_maj_axis_len
	y_start = y_cent - np.cos(orientation) * semi_maj_axis_len

	x_end = x_cent + np.sin(orientation) * semi_maj_axis_len
	y_end = y_cent + np.cos(orientation) * semi_maj_axis_len

	boxes = np.array([x_start, y_start, x_end, y_end])
	return boxes.astype(int)


def get_rerecist_coords(mask:MedImage) -> np.array:
	"""Get the RERECIST coordinates for a mask as the corners of an oriented bounding box"""
	# Convert the sitk.Image to a numpy array
	np_mask = mask.to_numpy()[0]
	# Sum the mask in the x and y axes to find the axial slice with the largest tumour area
	axial_sum = np.sum(np_mask, axis=(1,2))
	# Get the index of the axial slice with the largest tumour area
	max_axial_index = np.argmax(axial_sum)

	max_slice = np_mask[max_axial_index]
	rerecist_coords = mask2D_to_oriented_bbox(max_slice)

	return rerecist_coords, max_axial_index


def image_proc(image_path:Path) -> MedImage:
	"""Process image for use in the AAuRA Benchmarking tool 
	
	Parameters
	----------
	image_path : Path
		Path to the image to process
	
	Returns
	-------
	MedImage
		Processed MedImage object, cast to Int32
	"""
	# Read in image
	image = sitk.ReadImage(str(image_path))
	# Cast image to Int16
	image = sitk.Cast(image, sitk.sitkInt32)
	# Convert to MedImage
	image = MedImage(image)

	return image


def mask_proc(mask_path):
	"""Process mask for use in the AAuRA Benchmarking tool

	Parameters
	----------
	mask_path : Path
		Path to the mask to process

	Returns
	-------
	Mask
		Processed Mask object, cast to UInt8
	"""
	# Read in mask
	mask = sitk.ReadImage(str(mask_path))
	# Cast mask to UInt8
	mask = sitk.Cast(mask, sitk.sitkUInt8)

	# TODO: Find the maximum pixel value, this will be the number of volumes, make a little roi mapping then go back to using VectorMask

	# Convert to MedImageTools Mask
	mask = Mask(mask, metadata={"mask.ndim": 3})

	return mask


def process_one(sample:pd.Series,
				dataset:str,
				timepoint:str):
	id = sample['File Name']
	logger.info(f'Processing sample: {id}')

	image_path = Path(dataset) / timepoint / 'images' / f'{id}_0000.nii.gz'
	mask_path = Path(dataset) / timepoint / 'labels' / f'{id}.nii.gz'

	# Process image
	image = image_proc(dirs.RAWDATA / image_path)
	mask = mask_proc(dirs.RAWDATA / mask_path)
	logger.info(f'Image and mask loaded for sample: {id}')

	if mask.volume_count > 1:
		logger.warning(f'Sample {id} has more than one volume in the mask. Current volume count: {mask.volume_count}')
		raise NotImplementedError('VectorMask processing not implemented yet.')
	
		#TODO: Multiple ROI handling
	else:		
		# Get RERECIST coords from mask
		logger.info(f'Extracting RERECIST coordinates from mask for sample: {id}')
		rerecist_coords, max_axial_index = get_rerecist_coords(mask)

	# Save out processed image and mask
	proc_path_stem = Path(dataset, "images", timepoint, id)
	proc_image_path = dirs.PROCDATA / proc_path_stem / 'CT.nii.gz'
	# TODO: update this when multiple mask handling implemented, need to label it as mask_0, mask_1, etc.
	proc_mask_path = dirs.PROCDATA / proc_path_stem / 'mask_0.nii.gz'

	if not proc_image_path.parent.exists():
		proc_image_path.parent.mkdir(parents=True, exist_ok=True)
	if not proc_mask_path.parent.exists():
		proc_mask_path.parent.mkdir(parents=True, exist_ok=True)
	
	sitk.WriteImage(image, str(proc_image_path))
	sitk.WriteImage(mask, str(proc_mask_path))
	logger.info(f'Processed image and mask saved for sample: {id}')

	sample_index = {"id": id,
					"image_path": proc_path_stem / 'CT.nii.gz',
					"mask_path": proc_path_stem / 'mask.nii.gz',
					"annotation_type": "RERECIST",
					"annotation_coords": rerecist_coords,
					"largest_slice_index": int(max_axial_index),
					"size": image.size.to_tuple(),
					"spacing": image.spacing.to_tuple(),
					"origin": image.origin.to_tuple(),
					"direction": image.direction.to_matrix(),
					"mask_volume": mask.fingerprint["sum"],
					"lesion_location": None,
					"source": None
					}

	return sample_index


def process(dataset:str,
			metadata_file:Path = None,
			anatomy_match_file:Path = None,
			drop_data:list = None
			) -> pd.DataFrame:
	"""Process the specified dataset for use in the AAuRA Benchmarking tool
	
	Parameters
	----------
	dataset : str
		Name of the dataset to process
	metadata_file : Path, optional
		Name of the metadata file to use, by default None. Should exist in the rawdata/dataset folder
	anatomy_match_file : Path, optional
		Name of the anatomy match file to use for the lesion location column of the output, by default None. Should exist in the rawdata/dataset folder
	drop_data : list, optional
		List of data sources to drop from the dataset, by default None.

	Returns
	-------
	pd.DataFrame
		DataFrame containing the dataset index
	"""
	logger.info(f'Processing dataset: {dataset}')

	# Load metadata file
	logger.info(f'Loading metadata file: {metadata_file}')
	metadata = pd.read_csv(dirs.RAWDATA / dataset / metadata_file)

	if drop_data is not None:
		logger.info(f'Dropping data from sources: {drop_data}')
		metadata = metadata[~metadata.Source.str.contains('|'.join(drop_data))]

	if anatomy_match_file is not None:
		logger.info(f'Using anatomy match file: {anatomy_match_file}')
		anatomy_match = pd.read_csv(dirs.RAWDATA / dataset / anatomy_match_file)

	timepoints = ['Baseline', 'Synthetic_Follow_Up']

	for timepoint in timepoints:
		logger.info(f'Processing timepoint: {timepoint}')

		try:
			dataset_index = [
				process_one(
					sample=sample,
					dataset=dataset,
					timepoint=timepoint
				)
				for _, sample in metadata.iterrows()
			]
		except Exception as e:
			message = 'Error processing image data.'
			logger.error(message)
			raise e
		
		# Convert dataset index to DataFrame
		dataset_index_df = pd.DataFrame(dataset_index)

		# Add the source and lesion location columns
		dataset_index_df['source'] = metadata['Source'].values

		if anatomy_match_file is not None:
			# Map lesion location using anatomy match file
			# If 'Source' value contains one of the 'Dataset' values, map the corresponding 'Anatomy' value
			for _, source_dataset in anatomy_match.iterrows():
				dataset_index_df.loc[dataset_index_df['source'].str.contains(source_dataset['Dataset']), 'lesion_location'] = source_dataset['Anatomy']

		# Save out dataset index
		index_save_path = dirs.PROCDATA / dataset / "images" / timepoint / f'{timepoint}_aaura_index.csv'
		dataset_index_df.to_csv(index_save_path, index=False)

	return dataset_index_df


if __name__ == '__main__':
	logger.info('Starting data processing for LesionLocator')
	process(dataset="CVPR_LesionLocator",
		    metadata_file=Path("naming_1.csv"),
			anatomy_match_file=Path("dataset_anatomy_match.csv"),
			drop_data=['coronacases'])
