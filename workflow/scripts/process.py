import logging
import pandas as pd

from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from damply import dirs

from utils_images import image_proc, mask_proc

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("lesion_locator_processing.log")
)

logger = logging.getLogger(__name__)


def process_one(sample:pd.Series,
				dataset:str,
				timepoint:str) -> dict:
	id = sample['File Name']
	logger.info(f'Processing sample: {id}')

	image_path = Path(dataset) / 'images' / timepoint / 'images' / f'{id}_0000.nii.gz'
	mask_path = Path(dataset) / 'images' / timepoint / 'labels' / f'{id}.nii.gz'

	proc_path_stem = Path(dataset, "images", timepoint, id)
	# Process image
	image_metadata = image_proc(dirs.RAWDATA / image_path, proc_path_stem)
	logger.info(f'Image loaded, processed, and saved for sample: {id}')

	try:
		masks_metadata = mask_proc(dirs.RAWDATA / mask_path, proc_path_stem)
		logger.info(f'Mask loaded, processed, and saved for sample: {id}')
	except ValueError as e:
		# If a sample isn't labeled, skip it
		message = f'Error processing mask for sample {id}: {e}. Will be skipped.'
		logger.error(message)
		return {}

	sample_index = {}
	for mask_key, mask_metadata in masks_metadata.items():
		sample_index[f"{id}_{mask_key}"] = {"id": id,
									  		"image_path": proc_path_stem / 'CT.nii.gz',
									  		"mask_path": proc_path_stem / f'mask_{mask_key}.nii.gz',
											"mask_idx": int(mask_key),
											"mask_voxel_label": int(mask_metadata["voxel_label"]),
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
			drop_data:list = None,
			append_index:bool = False,
			disease_site:str | None = None,
			parallel:bool = False,

			n_jobs:int = -1
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
	append_index : bool, optional
		Whether to append to the existing index (e.g. new data has been processed). If False, will overwrite existing index file.
	disease_site : str, optional
		Name of disease site file data is located in if organized that way in data/rawdata. 
		Ex. data/procdata/MultiSite/CVPR_LesionLocator, would have this set to 'MultiSite'.
		Will be appended to dirs.RAWDATA and dirs.PROCDATA for length of the run.
	parallel : bool, optional
		Whether to run processing in parallel, default is False.
	n_jobs : int, optional
		Number of jobs to run in parallel. Parallel must be set to True for this to be used. Default is -1.

	Returns
	-------
	pd.DataFrame
		DataFrame containing the dataset index
	"""
	logger.info(f'Processing dataset: {dataset}')

	# Set up data dirs with disease site if included
	dirs.RAWDATA = dirs.RAWDATA / disease_site if disease_site else dirs.RAWDATA
	dirs.PROCDATA = dirs.PROCDATA / disease_site if disease_site else dirs.PROCDATA

	# Load metadata file
	logger.info(f'Loading metadata file: {metadata_file}')
	metadata = pd.read_csv(dirs.RAWDATA / dataset / metadata_file)

	if drop_data is not None:
		logger.info(f'Dropping data from sources: {drop_data}')
		metadata = metadata[~metadata.Source.str.contains('|'.join(drop_data))]

	if anatomy_match_file is not None:
		logger.info(f'Using anatomy match file: {anatomy_match_file}')
		anatomy_match = pd.read_csv(dirs.RAWDATA / dataset / anatomy_match_file)

	timepoints = ['Baseline']#, 'Synthetic_Follow_Up']

	for timepoint in timepoints:
		logger.info(f'Processing timepoint: {timepoint}')

		dataset_index = {}
		try:
			# Parallel processing
			if parallel:
				dataset_index_list = Parallel(n_jobs=n_jobs)(
					delayed(process_one)(
						sample=sample,
						dataset=dataset,
						timepoint=timepoint
						)
						for _, sample in tqdm(
							metadata.iterrows(), 
							desc=f"Processing {timepoint} images for AAuRA...", 
							total=len(metadata)
						)
				)

				dataset_index.update(sample_metadata for sample in dataset_index_list for sample_metadata in sample.items())

			else:
			# Sequential processing
				for _, sample in tqdm(metadata.iterrows(), desc=f"Processing {timepoint} images for AAuRA...", total=len(metadata)):
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

		if anatomy_match_file is not None:
			# Map lesion location using anatomy match file
			# If 'Source' value contains one of the 'Dataset' values, map the corresponding 'Anatomy' value
			for _, source_dataset in anatomy_match.iterrows():
				dataset_index_df.loc[dataset_index_df['source'].str.contains(source_dataset['Dataset']), 'lesion_location'] = source_dataset['Anatomy']

		# Set up output path for index file
		index_save_path = dirs.PROCDATA / dataset / "images" / timepoint / f'aaura_{timepoint}_index.csv'
		if not index_save_path.parent.exists():
			index_save_path.parent.mkdir(parents=True, exist_ok=True)

		# Check if index file already exists
		if index_save_path.exists():
			logger.info(f'Index file already exists at: {index_save_path}')
			if append_index:
				logger.info(f'Appending to existing index file at: {index_save_path}')
				# Load in the existing index file
				existing_index_df = pd.read_csv(index_save_path)
				# Append new index to existing index, keeping new index entries
				dataset_index_df = pd.concat([existing_index_df, dataset_index_df], ignore_index=True)
				# Drop duplicate entries based on id, image_path, and mask_idx, keeping the last occurrence
				dataset_index_df = dataset_index_df.drop_duplicates(subset=['id','image_path','mask_path','mask_idx'],keep='last', ignore_index=True)
				# Sort the index by id and mask_idx
				dataset_index_df = dataset_index_df.sort_values(by=['id', 'mask_idx'], ignore_index=True)

			else:
				# If append is not specified, will overwrite existing index file
				logger.info(f'Overwriting existing index file at: {index_save_path}')

		# Save out the index file
		dataset_index_df.to_csv(index_save_path, index=False)

	return dataset_index_df


if __name__ == '__main__':
	logger.info('Starting data processing for LesionLocator')
	process(dataset="CVPR_LesionLocator",
		    metadata_file=Path("images/naming_3.csv"),
			anatomy_match_file=Path("metadata/dataset_anatomy_match.csv"),
			drop_data=['coronacases','NIH-LYMPH'],
			append_index=False,
			disease_site='MultiSite',
			parallel=True,
			n_jobs=-1)
