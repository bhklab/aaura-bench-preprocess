import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from damply import dirs
from imgtools.coretypes import Mask, MedImage
from imgtools.transforms.functional import bias_correction
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def get_max_area_slice(mask:MedImage) -> tuple[np.array, int]:
	"""Get the slice and index of the axial slice with the largest ROI area in a mask
	
	Parameters
	----------
	mask: MedImage
		The mask to analyze, expected to be a binary mask with shape (z, x, y)
	
	Returns
	-------
	max_area_slice: np.array
		The 2D slice with the largest ROI area
	max_axial_index: int
		The index of the axial slice with the largest ROI area
	"""
	# Convert the sitk.Image to a numpy array
	np_mask = mask.to_numpy()[0]
	# Sum the mask in the x and y axes to find the axial slice with the largest tumour area
	axial_sum = np.sum(np_mask, axis=(1,2))
	# Get the index of the axial slice with the largest tumour area
	max_axial_index = np.argmax(axial_sum)
	# Select out the slice with the largest index
	max_area_slice = np_mask[max_axial_index]

	return max_area_slice, max_axial_index



def get_centered_bbox(center_pt: np.array, 
                      major_axis_length: float, 
                      max_axial_index: int) -> np.ndarray: 
    '''  
    Get a square bounding box centered on the midpoint of the RERECIST line. 

    Parameters
    ----------
    center_pt: np.array
        Contains the midpoint of the RERECIST line in [x_cent, y_cent] form.
    major_axis_length: float
        Length of the RERECIST line.
    max_axial_index: int
        The index of the slice with the largest area.

    Returns
    -----------
    centered_bbox_3d: np.ndarray
        A bounding box with shape compatible with nnInteractive prompt input. 
        NOTE: In readme.md, bounding boxes are defined as [[x1, x2], [y1, y2], [z1, z2]], 
        but in my experience they are actually [[y1, y2], [x1, x2], [z1, z2]]. The latter
        is what is implemented here.
    '''
    #Get top left and bottom right corners of the bounding box 
    x_tl = center_pt[0] - major_axis_length / 2 
    y_tl = center_pt[1] - major_axis_length / 2 

    x_br = center_pt[0] + major_axis_length / 2 
    y_br = center_pt[1] + major_axis_length / 2

    # Note that the nnInteractive examples say the bounding box is in (x, y, z) order, but
    # in my experience, it actually expects (y, x, z) order 
    centered_bbox_3d = np.array([min(512, int(y_tl)), min(512, int(y_br)), min(512, int(x_tl)), min(512, int(x_br)), int(max_axial_index), int(max_axial_index + 1)])

    return centered_bbox_3d


def mask3D_to_centered_bbox(mask:MedImage,  # noqa
							max_axial_index:int = None
							) -> np.array:
	"""Convert a 3D binary mask to a centered bounding box around the region of interest
	
	Parameters
	----------
	mask: MedImage
		The mask to analyze, expected to be a binary mask with shape (z, x, y)
	max_axial_index: int, optional
		The index of the axial slice to use for calculating the bounding box. If None, the slice with the largest tumour area will be used. Default is None.
	
	Returns
	-------
	centered_bbox_2d: np.array
		A bounding box with shape compatible with nnInteractive prompt input, centered on the RERECIST line. 
		NOTE: In readme.md, bounding boxes are defined as [[x1, x2], [y1, y2], [z1, z2]], but in my experience they are actually [[y1, y2], [x1, x2], [z1, z2]]. The latter is what is implemented here.

	"""

	if max_axial_index is None:
		# Get the index of the axial slice with the largest tumour area
		max_area_slice, max_axial_index = get_max_area_slice(mask)
	else:
		max_area_slice = mask.to_numpy()[0][max_axial_index]

	# Get the centroid and major axis length of the region in the slice with the largest tumour area - this is the centroid and major axis length of the RERECIST line
	props = regionprops(max_area_slice)[0]
	y_cent, x_cent = props.centroid
	maj_axis_len = props.axis_major_length

	# Pass the center point of the RECIST line in the expected format 
	center_pt = np.array([x_cent, y_cent])
	centered_bbox_3d = get_centered_bbox(center_pt, maj_axis_len, max_axial_index)

	return centered_bbox_3d	



def mask2D_to_oriented_bbox(mask:np.array) -> np.array:  # noqa
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
	# Get the index of the axial slice with the largest tumour area
	max_slice, max_axial_index = get_max_area_slice(mask)

	rerecist_coords = mask2D_to_oriented_bbox(max_slice)

	return rerecist_coords, max_axial_index



def mr_proc(scan:sitk.Image) -> sitk.Image:
	"""Apply bias correction to MR image"""
	# sitk N4 bias correction requires the image to be a float

	logging.info("Starting bias correction...")
	scan_float = sitk.Cast(scan, sitk.sitkFloat32)

	return bias_correction(scan_float)



def scan_proc(scan_path:Path,
			   proc_path_stem:str|None = None,
			   modality:str = 'CT') -> dict:
	"""Process scan for use in the AAuRA Benchmarking tool 
	
	Parameters
	----------
	scan_path : Path
		Path to the scan to process
	proc_path_stem : str
		Path to add to dirs.PROCDATA to save scan out to.
	
	Returns
	-------
	MedImage
		Processed MedImage object, CT's cast to Int32, MR's to Float32
	"""
	# Read in scan
	scan_sitk = sitk.ReadImage(str(scan_path))

	if modality == 'CT':
		# Cast scan to Int16
		scan_sitk = sitk.Cast(scan_sitk, sitk.sitkInt32)
	
	elif modality == 'MR':
		# Apply bias correction and cast to Float32
		scan_sitk = mr_proc(scan_sitk)
		logging.info("Finished bias correction")

	# Convert to MedImage
	scan = MedImage(scan_sitk)

	# Get scan metadata
	scan_metadata = scan.fingerprint

	# Convert size, spacing, origin, direction to tuples/lists for JSON serialization
	scan_metadata["size"] = scan.size.to_tuple()
	scan_metadata["spacing"] = scan.spacing.to_tuple()
	scan_metadata["origin"] = scan.origin.to_tuple()
	scan_metadata["direction"] = scan.direction.to_matrix()

	# Save out transformed scan
	if proc_path_stem is not None:
		proc_scan_stem = proc_path_stem / f'{modality}.nii.gz'
		proc_scan_path = dirs.PROCDATA / proc_scan_stem
		if not proc_scan_path.parent.exists():
			proc_scan_path.parent.mkdir(parents=True, exist_ok=True)
		sitk.WriteImage(scan, str(proc_scan_path))
		logger.info(f'Processed scan saved at: {proc_scan_path}')
		scan_metadata["scan_path"] = proc_scan_stem

	return scan_metadata


def mask_proc(mask_path:Path,
			  proc_path_stem:str|None = None) -> dict:
	"""Process mask for use in the AAuRA Benchmarking tool

	Parameters
	----------
	mask_path : Path
		Path to the mask to process
	proc_path_stem : str
		Path to add to dirs.PROCDATA to save mask out to.

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
		message = f'Mask at {mask_path} has no labelled volumes.'
		logger.info(message)
		raise ValueError(message)
	
	else:
		proc_mask_metadata = {}
		logger.info(f'Mask at {mask_path} has {len(unique_labels)-1} labelled volumes.')

		for volume_idx in range(1, len(unique_labels)):
			# Get the label value for the current volume (won't necessarily be equal to volume_idx)
			volume_label = unique_labels[volume_idx]
			# Extract the volume with the current label (volume_idx)
			idx_mask = (label_array == (volume_label)).astype(np.uint8)

			# Convert the extracted volume back to a sitk.Image
			idx_mask_sitk = sitk.GetImageFromArray(idx_mask)
			# Copy the metadata from the original mask
			idx_mask_sitk.CopyInformation(mask)
			# Convert to MedImageTools Mask
			idx_mask_mi = Mask(idx_mask_sitk, metadata={"mask.ndim": 3})
			idx_mask_metadata = idx_mask_mi.fingerprint
			idx_mask_metadata["voxel_label"] = int(volume_label)
			
			# Get RERECIST coords for current volume
			rerecist_coords, max_axial_index = get_rerecist_coords(idx_mask_mi)
			idx_mask_metadata["annotation_coords"] = rerecist_coords
			idx_mask_metadata["largest_slice_index"] = int(max_axial_index)
			
			# Get a bounding box centered on the RERECIST line for current volume
			centered_bbox = mask3D_to_centered_bbox(idx_mask_mi, max_axial_index=max_axial_index)
			idx_mask_metadata["centered_bbox_coords"] = centered_bbox

			# Write out the individual mask volume
			if proc_path_stem is not None:
				proc_mask_stem = proc_path_stem / f'mask_{volume_idx}.nii.gz'
				proc_mask_path = dirs.PROCDATA / proc_mask_stem
				if not proc_mask_path.parent.exists():
					proc_mask_path.parent.mkdir(parents=True, exist_ok=True)
				sitk.WriteImage(idx_mask_sitk, str(proc_mask_path))
				logger.info(f'Processed mask volume {volume_idx} saved at: {proc_mask_path}')
				idx_mask_metadata["mask_path"] = proc_mask_stem

			proc_mask_metadata[f"{volume_idx}"] = idx_mask_metadata

		return proc_mask_metadata