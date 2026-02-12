import logging
from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from damply import dirs
from imgtools.coretypes import Mask
from pydanclick import from_pydantic
from utils_images import get_rerecist_coords, mask3D_to_centered_bbox
from utils_index import insert_SampleID, make_edges_df

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("mit_to_aaura_index.log")
)

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration for converting MIT index to aaura index."""

    datasource: str = Field(..., description="The source of the dataset (e.g. TCIA)")
    dataset: str = Field(..., description="The name of the dataset (e.g. RADCURE)")
    ROI_key: str = Field(default=".*", description="The key in the MIT index that identifies the ROI masks of interest")
    image_modality: str = Field(default="CT", description="The modality of the images (e.g. CT)")
    mask_modality: str = Field(default="RTSTRUCT", description="The modality/modalities of the masks (e.g. RTSTRUCT)")
    disease_site: Optional[str|None] = Field(default="", description="The disease site subdirectory in raw data (e.g. HeadNeck)")
    special_prefix: Optional[str|None] = Field(default="", description="A special prefix for the dataset name")
    special_suffix: Optional[str|None] = Field(default="", description="A special suffix for the dataset name")

    # @field_validator('mask_modality', mode='before')
    # @classmethod
    # def convert_to_list(cls, v):
    #     """Convert a comma-separated string to a list."""
    #     if isinstance(v, str):
    #         return [item.strip() for item in v.split(',')]
    #     return v

    @field_validator('special_prefix', 'special_suffix', mode='before')
    @classmethod
    def set_empty_string_defaults(cls, v):
        """Convert None to empty string."""
        return v or ""


class AAuraIndexRow(BaseModel):
	"""Represents a single row in the aaura index."""
	
	id: str
	image_path: Path
	mask_path: Path
	mask_idx: int = 1
	mask_voxel_label: int = 1
	size: tuple
	spacing: tuple
	origin: tuple
	direction: tuple
	mask_volume: float
	annotation_type: str = "RERECIST"
	annotation_coords: dict
	largest_slice_index: int
	centered_bbox_coords: dict
	lesion_location: Optional[str] = None
	source: str



def mit_to_aaura_index(config: DatasetConfig) -> pd.DataFrame:
    """Converts the MIT index for a given dataset into an aaura-compatible index.
    
    Parameters
    ---------- 
    config: MitIndexConfig
        Configuration object containing parameters for the conversion

    Returns
    -------
    pd.DataFrame
        The aaura-compatible index

    """
    datasource = config.datasource
    dataset = config.dataset
    ROI_key = config.ROI_key
    image_modality = config.image_modality
    mask_modality = config.mask_modality
    disease_site = config.disease_site
    special_prefix = config.special_prefix
    special_suffix = config.special_suffix

    if "," in mask_modality:
        mask_modality = [item.strip() for item in mask_modality.split(',')]
    else:
        mask_modality = [mask_modality]

    if disease_site:
        # Set up data dirs with disease site if included
        dirs.RAWDATA = dirs.RAWDATA / disease_site

    dataset_path_prefix = Path(f"{datasource}_{dataset}/images/mit_{dataset}{special_suffix}")
    mit_dir_path = dirs.RAWDATA / dataset_path_prefix

    mit_index = pd.read_csv(mit_dir_path / f"mit_{special_prefix}{dataset}{special_suffix}_index-simple.csv")

    # Select out mask rows based on the ROI key specified
    mit_index_mask_rows = mit_index[mit_index['matched_rois'].str.contains(ROI_key, case=False, na=False)]

    # Get the rows for the images referenced by the ROI masks
    mit_index_image_rows = mit_index[mit_index['SeriesInstanceUID'].isin(mit_index_mask_rows['ReferencedSeriesUID'])]

    # Concatenate the image and mask rows into a single dataframe
    selected_index_rows = pd.concat([mit_index_mask_rows, mit_index_image_rows], ignore_index=True)
    # Create a SampleID row for indexing purposes
    selected_index_rows = insert_SampleID(selected_index_rows)

    mod_dfs = {}
    for mod in mask_modality:
        # Rearrange the dataframe so each row has a mask and image pair
        modality_edges_df = make_edges_df(selected_index_rows, 
                                    image_modality=image_modality,
                                    mask_modality=str(mod))
        mod_dfs[mod] = modality_edges_df

    # Combine the dataframes for each mask modality
    matched_index_rows = pd.concat(mod_dfs.values(), ignore_index=True)
    # Sort the index rows by SampleID
    matched_index_rows = matched_index_rows.sort_values('SampleID_image')

    if matched_index_rows.empty:
        message = "No matching image and masks found with current settings. No aaura index to generate."
        raise ValueError(message)

    # Get the columns out of mit_index needed in the aaura index
    aaura_columns_dict = {"id":matched_index_rows['SampleID_image'],
        "image_path": dataset_path_prefix / matched_index_rows['filepath_image'],
        "mask_path": dataset_path_prefix / matched_index_rows['filepath_mask'],
        "size": matched_index_rows["size_image"],
        "spacing": matched_index_rows["spacing_image"],
        "origin": matched_index_rows["origin_image"],
        "direction": matched_index_rows["direction_image"],
        "mask_volume": matched_index_rows["sum_mask"],
        }

    aaura_index = pd.DataFrame.from_dict(aaura_columns_dict)
    aaura_index['lesion_location'] = disease_site.lower()
    aaura_index['source'] = f"{special_prefix}{dataset}"
    aaura_index.insert(3, 'mask_idx', 1)
    aaura_index.insert(4, 'mask_voxel_label', 1)

    annotation_coords = {}
    largest_slice_index = {}
    centered_bbox_coords = {}
    for sample_index, sample in aaura_index.iterrows():
        logger.info(f"Processing sample: {sample['id']}")
        # Load in Mask as MedImageTools Mask object
        mask = Mask.from_file(dirs.RAWDATA / sample['mask_path'], metadata={"mask.ndim": 3})
        # Get RERECIST coords for current volume
        rerecist_coords, max_axial_index = get_rerecist_coords(mask)

        annotation_coords[sample_index] = rerecist_coords
        largest_slice_index[sample_index] = int(max_axial_index)
        centered_bbox_coords[sample_index] = mask3D_to_centered_bbox(mask, max_axial_index=max_axial_index)

    aaura_index.insert(5, 'annotation_type', 'RERECIST')
    aaura_index.insert(6, 'annotation_coords', annotation_coords)
    aaura_index.insert(7, 'largest_slice_index', largest_slice_index)
    aaura_index.insert(8, 'centered_bbox_coords', centered_bbox_coords)

    aaura_index.to_csv(mit_dir_path / f"aaura_{special_prefix}{dataset}{special_suffix}_index.csv", index=False)

    return aaura_index

@click.command()
@from_pydantic(DatasetConfig)
def run_mit_to_aaura_index(dataset_config: DatasetConfig):
    """Run the mit_to_aaura_index function with example configuration."""
    aaura_index = mit_to_aaura_index(config=dataset_config)
    # print(aaura_index.head())


if __name__ == "__main__":
    run_mit_to_aaura_index()