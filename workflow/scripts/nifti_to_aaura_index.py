from typing import Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo

import logging
from pathlib import Path

import click
import pandas as pd
from damply import dirs
from imgtools.pattern_parser import PatternResolver
from joblib import Parallel, delayed
from pydanclick import from_pydantic
from tqdm import tqdm
from utils_images import scan_proc, mask_proc
from utils_models import AAuraIndexRow

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("nifti_to_aaura_index.log")
)

logger = logging.getLogger(__name__)

class NiftiDatasetConfig(BaseModel):
    """Configuration for nifti image dataset to create aaura index for."""

    datasource: str = Field(..., description="The source of the dataset (e.g. TCIA, PMCC)")
    dataset: str = Field(..., description="The name of the dataset (e.g. RADCURE)")
    patient_id_col: str = Field(default="patient_id", description="The name of the column in the metadata files that contains the patient ID")
    scan_name_pattern: str = Field(default="{patient_id}.nii.gz", description="The pattern to use for generating filenames in the index. Use {patient_id} and {series_uid} as placeholders for the patient ID and series UID, respectively.")
    mask_name_pattern : Optional[str] = Field(default=None, validate_default=True,description="The pattern to use for generating mask filenames in the index. Use {patient_id} and {series_uid} as placeholders for the patient ID and series UID, respectively.")
    scan_path_pattern: Optional[str] = Field(default="scans", description="The pattern to use for generating filepaths to the scan images starting after {datasource}_{dataset}/images.")
    mask_path_pattern: Optional[str] = Field(default="masks", description="The pattern to use for generating filepaths to the segmentation masks starting after {datasource}_{dataset}/images.")
    disease_site: Optional[str] = Field(default=None, description="The disease site of the dataset (e.g. lung)")
    metadata_files: Optional[list[str]] = Field(default=[], description="List of paths to metadata files to include in the index") 
    image_modality: Optional[str] = Field(default="CT", description="The modality of the images (e.g. CT)")
    drop_data: Optional[dict[str, list[str]]] = Field(default=None, description="Dictionary specifying any data to drop from the index (e.g. {'patient_id': ['TCGA-02-0047']})")
    anatomy_match_file: Optional[str] = Field(default=None, description="Path to a separate metadata file containing mappings of sources to anatomy labels, for use when dataset contains multiple other datasets with different disease sites.")
    sample_id_pattern: Optional[str] = Field(default=None, description="Pattern to use to build a sample_id column to use instead of patient_id_col. (e.g. '{dataset}_{patient_id}')")

    @field_validator('scan_name_pattern')
    def validate_filename_pattern(cls, v):
        if ".nii.gz" not in v:
            return f"{v}.nii.gz"
        return v
    
    @field_validator('mask_name_pattern')
    def validate_maskname_pattern(cls, v, info: ValidationInfo):
        if v is not None and ".nii.gz" not in v:
            return f"{v}.nii.gz"
        elif v is None:
            v = info.data['scan_name_pattern']
        return v

    @field_validator('metadata_files')
    @classmethod
    def validate_metadata_files(cls, v: list[str], info: ValidationInfo):
        if not isinstance(v, list):
            raise ValueError("metadata_files must be a list")
        elif len(v) == 0:
            print("No metadata files specified. Attempting to find metadata files in expected location...")
            metadata_dir_path = dirs.RAWDATA / f"{info.data['datasource']}_{info.data['dataset']}/metadata"
            v = list(metadata_dir_path.glob(r"*.csv")) + list(metadata_dir_path.glob(r"*.json"))
            if len(v) == 0:
                raise ValueError(f"No metadata files found in expected location: {metadata_dir_path}")
            if info.data['anatomy_match_file'] is not None:
                v.drop(info.data['anatomy_match_file'], inplace=True)
        return v
    

#TODO: implement this function
def combine_all_metadata(metadata_files: list[Path],
                         key_name:str = 'patient_id') -> pd.DataFrame:
    
    pass


def image_path_resolver(metadata_row:pd.Series,
                        image_path_pattern:str,
                        image_file_pattern:str) -> str:
    
    filename_format = f"{image_path_pattern}/{image_file_pattern}"

    image_path_resolver = PatternResolver(filename_format)

    return image_path_resolver.resolve(metadata_row.to_dict())


def sample_id_resolver(metadata_row:pd.Series,
                       sample_id_pattern:str):
    sample_id_resolver = PatternResolver(sample_id_pattern)

    return sample_id_resolver.resolve(metadata_row.to_dict())


def metadata_setup(metadata_df:pd.DataFrame,
                   config):
    
    dataset_name = f"{config.datasource}_{config.dataset}"
    drop_data = config.drop_data
    patient_id_col = config.patient_id_col
    sample_id_pattern = config.sample_id_pattern
    scan_path_pattern = config.scan_path_pattern
    scan_name_pattern = config.scan_name_pattern
    mask_path_pattern = config.mask_path_pattern
    mask_name_pattern = config.mask_name_pattern

    # Handle any data needing to be removed before processing
    if drop_data is not None:
        logger.info(f'Dropping data from: {drop_data}')
        for column_name, values in drop_data.items():
            metadata_df = metadata_df[metadata_df[column_name].str.contains('|'.join(values))]

    metadata_df['raw_scan_path'] = metadata_df.apply(
        lambda row: Path(f"{dataset_name}") / "images" / image_path_resolver(row, scan_path_pattern, scan_name_pattern), 
        axis = 1
        )

    metadata_df['raw_mask_path'] = metadata_df.apply(
        lambda row: Path(f"{dataset_name}") / "images" / image_path_resolver(row, mask_path_pattern, mask_name_pattern), 
        axis = 1
        )
    
    # Handle setting up the sample_id column by matching a pattern or copying the patient_id_col
    if sample_id_pattern is not None:
        sample_id_col = metadata_df.apply(lambda row: sample_id_resolver(row, sample_id_pattern))
    else:
        sample_id_col = metadata_df[patient_id_col]
    
    metadata_df.insert(loc=0,
                       column='sample_id',
                       value=sample_id_col
                      )
    
    # Rename the patient_id_col to source_patient_id
    metadata_df = metadata_df.rename(columns = {patient_id_col: "source_patient_id"})

    return metadata_df



def process_one(sample:pd.Series,
                proc_path_stem: Path,
                config:NiftiDatasetConfig
                ) -> dict[dict]:
    sample_id = sample['sample_id']

    logger.info(f'Processing sample: {sample_id}')

    # Add the filename to the end of the proc_path_stem
    proc_sample_path_stem = proc_path_stem / sample_id

    # Process scan
    scan_metadata = scan_proc(scan_path = dirs.RAWDATA / sample['raw_scan_path'],
                              proc_path_stem=proc_sample_path_stem,
                              modality = config.image_modality)
    logger.info(f'Image loaded, processed, and saved for sample: {sample_id}')

    try:
        masks_metadata = mask_proc(mask_path = dirs.RAWDATA / sample['raw_mask_path'],
                                   proc_path_stem=proc_sample_path_stem)
        logger.info(f'Mask loaded, processed, and saved for sample: {sample_id}')
    except ValueError as e:
        # If a sample isn't labeled, skip it
        message = f'Error processing mask for sample {sample_id}: {e}. Will be skipped.'
        logger.exception(message)
        return {}

    sample_index = {}
    for mask_key, mask_metadata in masks_metadata.items():
        sample_index[f"{sample_id}_{mask_key}"] = {"id": sample_id,
                                                   "scan_path": scan_metadata["scan_path"],
                                                   "mask_path": mask_metadata["mask_path"],
                                                   "mask_idx": int(mask_key),
                                                   "mask_voxel_label": int(mask_metadata["voxel_label"]),
                                                   "annotation_type": "RERECIST",
                                                   "annotation_coords": mask_metadata["annotation_coords"],
                                                   "largest_slice_index": mask_metadata["largest_slice_index"],
                                                   "centered_bbox_coords": mask_metadata["centered_bbox_coords"],
                                                   "size": scan_metadata["size"],
                                                   "spacing": scan_metadata["spacing"],
                                                   "origin": scan_metadata["origin"],
                                                   "direction": scan_metadata["direction"],
                                                   "mask_volume": mask_metadata["sum"],
                                                   "disease_site": config.disease_site,
                                                   "source_patient_id": sample["source_patient_id"]
                                            }
    return sample_index



def nifti_to_aaura_index(config: NiftiDatasetConfig,
                         append_index:bool = False,
                         parallel:bool = False,
                         n_jobs:int = -1
                         ) -> pd.DataFrame:
    """Processes a dataset of NifTi files into an AAuRA-compatible dataset and index.
    
    Parameters
    ----------
    config: NiftiDatasetConfig
        Configuration object containing parameters for the conversion

    Returns
    -------
    pd.DataFrame
        The AAuRA compatible index
    """
    dataset_name = f"{config.datasource}_{config.dataset}"
    metadata_files = config.metadata_files

    logger.info(f'Processing dataset: {dataset_name}')

    # Load metadata
    # Just handling CSV for now, will need to add other modalities
    metadata_df = pd.read_csv(dirs.RAWDATA / dataset_name /  "metadata" / metadata_files[0])
    metadata_df = metadata_setup(metadata_df, config)

    # Set up output for processed images and index
    proc_path_stem = Path(dataset_name, "images", f"aaura_{config.dataset}")
    aaura_index = {}
    try:
        if parallel:
        # Parallel processing
            aaura_index_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(process_one)(
                    sample=sample,
                    proc_path_stem=proc_path_stem,
                    config=config
                )
                for _, sample in tqdm(
                    metadata_df.iterrows(),
                    desc=f"Processing images for AAuRA index...",
                    total=len(metadata_df)
                )
            )

            aaura_index.update(sample_metadata for sample in aaura_index_list for sample_metadata in sample.items())

        else:
        # Sequential processing
            for _, sample in tqdm(
                metadata_df.iterrows(),
                desc=f"Processing images for AAuRA index...",
                total=len(metadata_df)
            ):
                aaura_index.update(process_one(sample=sample,
                                               proc_path_stem=proc_path_stem,
                                               config=config))
                
    except Exception:
        message = 'Error processing image data.'
        logger.exception(message)
        raise

    aaura_index_df = pd.DataFrame.from_dict(aaura_index, orient='index')

    # Set up output for index file
    index_save_path = dirs.PROCDATA / proc_path_stem / f'aaura_{config.dataset}_index.csv'
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

    aaura_index_df.to_csv(index_save_path, index=False)

    return aaura_index_df


# @click.command()
# @from_pydantic(NiftiDatasetConfig)
def run_nifti_to_aaura_index(dataset_config:NiftiDatasetConfig):
    aaura_index = nifti_to_aaura_index(dataset_config,
                                       parallel=True,
                                       n_jobs = -1)


if __name__ == "__main__":
    mama_mia_config = NiftiDatasetConfig(
        datasource='BCN-AIM',
        dataset='MAMA-MIA',
        patient_id_col = 'patient_id',
        scan_name_pattern="{patient_id}_0000.nii.gz",
        mask_name_pattern="{patient_id}.nii.gz",
        scan_path_pattern="scans/{patient_id}",
        mask_path_pattern="masks",
        metadata_files = ['clinical_and_imaging_info.csv'],
        image_modality="MR",
        disease_site='breast'
    )

    run_nifti_to_aaura_index(mama_mia_config)

