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


def image_proc(image_path:Path,
			   proc_path_stem:str|None = None) -> dict:
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

	if proc_path_stem is not None:
		proc_image_path = dirs.PROCDATA / proc_path_stem / 'CT.nii.gz'
		if not proc_image_path.parent.exists():
			proc_image_path.parent.mkdir(parents=True, exist_ok=True)
		sitk.WriteImage(image, str(proc_image_path))
		logger.info(f'Processed image saved at: {proc_image_path}')

	# Get image metadata
	image_metadata = image.fingerprint
	# Convert size, spacing, origin, direction to tuples/lists for JSON serialization
	image_metadata["size"] = image.size.to_tuple()
	image_metadata["spacing"] = image.spacing.to_tuple()
	image_metadata["origin"] = image.origin.to_tuple()
	image_metadata["direction"] = image.direction.to_matrix()

	return image_metadata


def mask_proc(mask_path:Path,
			  proc_path_stem:str|None = None) -> dict:
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

	# Convert mask to numpy array to check unique labels
	label_array = sitk.GetArrayFromImage(mask)
	unique_labels = np.unique(label_array)

	if len(unique_labels) == 1:
		logger.info(f'Mask at {mask_path} has no labelled volumes.')
		raise ValueError('Mask has no labelled volumes.')
	
	else:
		proc_mask_metadata = {}
		logger.info(f'Mask at {mask_path} has {len(unique_labels)-1} labelled volumes.')
		for volume_idx in range(1, len(unique_labels)):
			# Extract the volume with the current label (volume_idx)
			idx_mask = (label_array == (volume_idx)).astype(np.uint8)

			# Convert the extracted volume back to a sitk.Image
			idx_mask_sitk = sitk.GetImageFromArray(idx_mask)
			# Copy the metadata from the original mask
			idx_mask_sitk.CopyInformation(mask)
			# Convert to MedImageTools Mask
			idx_mask_mi = Mask(idx_mask_sitk, metadata={"mask.ndim": 3})
			idx_mask_metadata = idx_mask_mi.fingerprint

			# Write out the individual mask volume
			if proc_path_stem is not None:
				proc_mask_path = dirs.PROCDATA / proc_path_stem / f'mask_{volume_idx}.nii.gz'
				if not proc_mask_path.parent.exists():
					proc_mask_path.parent.mkdir(parents=True, exist_ok=True)
				sitk.WriteImage(idx_mask_sitk, str(proc_mask_path))
				logger.info(f'Processed mask volume {volume_idx} saved at: {proc_mask_path}')
			
			# Get RERECIST coords for current volume
			rerecist_coords, max_axial_index = get_rerecist_coords(idx_mask_mi)
			idx_mask_metadata["annotation_coords"] = rerecist_coords
			idx_mask_metadata["largest_slice_index"] = int(max_axial_index)

			proc_mask_metadata[f"{volume_idx}"] = idx_mask_metadata

		return proc_mask_metadata


def process_one(sample:pd.Series,
				dataset:str,
				timepoint:str):
	id = sample['File Name']
	logger.info(f'Processing sample: {id}')

	image_path = Path(dataset) / timepoint / 'images' / f'{id}_0000.nii.gz'
	mask_path = Path(dataset) / timepoint / 'labels' / f'{id}.nii.gz'

	proc_path_stem = Path(dataset, "images", timepoint, id)
	# Process image
	image_metadata = image_proc(dirs.RAWDATA / image_path, proc_path_stem)
	masks_metadata = mask_proc(dirs.RAWDATA / mask_path, proc_path_stem)
	logger.info(f'Image and mask loaded for sample: {id}')

	sample_index = {}
	for mask_key, mask_metadata in masks_metadata.items():
		sample_index[f"{id}_{mask_key}"] = {"id": id,
									  		"image_path": proc_path_stem / 'CT.nii.gz',
									  		"mask_path": proc_path_stem / f'mask_{mask_key}.nii.gz',
											"mask_idx": int(mask_key),
									  		"annotation_type": "RERECIST",
											"annotation_coords": mask_metadata["annotation_coords"],
											"largest_slice_index": mask_metadata["largest_slice_index"],
											"size": image_metadata["size"],
											"spacing": image_metadata["spacing"],
											"origin": image_metadata["origin"],
											"direction": image_metadata["direction"],
											"mask_volume": mask_metadata["sum"],
											"lesion_location": None,
											"source": sample['Source']
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

		dataset_index = {}
		try:
			for _, sample in metadata.iterrows():
				dataset_index.update(process_one(sample=sample,
												 dataset=dataset,
												 timepoint=timepoint)
									)					
		except Exception as e:
			message = 'Error processing image data.'
			logger.error(message)
			raise e
		
		# Convert dataset index to DataFrame
		dataset_index_df = pd.DataFrame.from_dict(dataset_index, orient='index')

		# Add the source and lesion location columns
		# dataset_index_df['source'] = metadata['Source'].values

		if anatomy_match_file is not None:
			# Map lesion location using anatomy match file
			# If 'Source' value contains one of the 'Dataset' values, map the corresponding 'Anatomy' value
			for _, source_dataset in anatomy_match.iterrows():
				dataset_index_df.loc[dataset_index_df['source'].str.contains(source_dataset['Dataset']), 'lesion_location'] = source_dataset['Anatomy']

		# Save out dataset index
		index_save_path = dirs.PROCDATA / dataset / "images" / timepoint / f'{timepoint}_aaura_index.csv'
		if not index_save_path.parent.exists():
			index_save_path.parent.mkdir(parents=True, exist_ok=True)
		dataset_index_df.to_csv(index_save_path, index=False)

	return dataset_index_df


if __name__ == '__main__':
	logger.info('Starting data processing for LesionLocator')
	process(dataset="CVPR_LesionLocator",
		    metadata_file=Path("naming_1.csv"),
			anatomy_match_file=Path("dataset_anatomy_match.csv"),
			drop_data=['coronacases'])
