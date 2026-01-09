import numpy as np
import pandas as pd
import SimpleITK as sitk

from imgtools.coretypes import MedImage, Mask, VectorMask
from joblib import Parallel, delayed
from pathlib import Path
from skimage.measure import regionprops
from tqdm import tqdm
from damply import dirs

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