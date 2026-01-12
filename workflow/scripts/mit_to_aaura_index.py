from damply import dirs
from pathlib import Path
import pandas as pd
import logging
from imgtools.coretypes import Mask

from utils_images import get_rerecist_coords
from utils_index import make_edges_df, insert_SampleID

logging.basicConfig(
	level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
	filename=dirs.LOGS / Path("mit_to_aaura_index.log")
)

logger = logging.getLogger(__name__)


datasource = "TCIA"
dataset = "RADCURE"
ROI_key = "GTVp"
image_modality = "CT"
mask_modality = ["RTSTRUCT"] # THIS HAS TO BE A LIST
lesion_location = "headneck"
special_prefix = "OCSCC_"
special_suffix = "_windowed"

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
aaura_index['lesion_location'] = lesion_location
aaura_index['source'] = f"{special_prefix}{dataset}"
aaura_index.insert(3, 'mask_idx', 1)
aaura_index.insert(4, 'mask_voxel_label', 1)

annotation_coords = {}
largest_slice_index = {}
for sample_index, sample in aaura_index.iterrows():
    print(sample['id'])
    mask = Mask.from_file(dirs.RAWDATA / sample['mask_path'], metadata={"mask.ndim": 3})
    # Get RERECIST coords for current volume
    rerecist_coords, max_axial_index = get_rerecist_coords(mask)

    annotation_coords[sample_index] = rerecist_coords
    largest_slice_index[sample_index] = int(max_axial_index)

aaura_index.insert(5, 'annotation_type', 'RERECIST')
aaura_index.insert(6, 'annotation_coords', annotation_coords)
aaura_index.insert(7, 'largest_slice_index', largest_slice_index)

aaura_index.to_csv(mit_dir_path / f"aaura_{special_prefix}{dataset}{special_suffix}_index.csv", index=False)