# Developer Notes

## Removing COVID-19 CT Lung from analysis
*2026-01-07*  
We chose to remove these samples from analysis as they do not have tumours present and only include a small number of samples. 

Did not include them in the dataset_anatomy_match.csv we made.

## Column setup for aaura index and med-imagetools processing
*2026-01-07*
For datasets that have been processed by med-imagetools, we will take in the index-simple.csv and extract the subset of columns we use for the aaura index.

Columns that will need to be calculated in addition are:

* annotation_type
* annotation_coords
* largest_slice_index
* lesion_location
* source (Optional, mostly used for datasets composed of multiple other datasets)

## Mask indexing for saving starting at 1 to reflect labels
*2026-01-07*  
Starting the mask labelling at 1 to reflect the voxel values they were in the original image. 0 is reserved for background.

## Add option to append new processed dataset index to existing index
*2026-01-08*
Today's solution chosen for handling what to do if processing new data but you want to preserve the already processed data.
`append_index` argument can be set such that an existing index file will be loaded in, the new processed data index will be concatenated to the end, checked for duplicates, sorted, and then saved. For duplicate checking, the new processed data entry will be kept. 

This way, if processing a dataset breaks in the middle, can update the metadata file with what data to process. Didn't want to implement image existence checking yet. 

## `mit_to_aaura_index` setup for datasets
*2026-01-12*

HCC-TACE-Seg  
```python
datasource = "TCIA"
dataset = "HCC-TACE-Seg"
ROI_key = "Mass"
image_modality = "CT"
mask_modality = ["SEG"]
lesion_location = "abdomen"
```  

4D-Lung
```python
datasource = "TCIA"
dataset = "4D-Lung"
ROI_key = "Tumor_"
image_modality = "CT"
mask_modality = ["RTSTRUCT"]
lesion_location = "abdomen"
```

RIDER-LungCT-Seg  
```python
datasource = "TCIA"
dataset = "RIDER-LungCT-Seg"
ROI_key = "GTVp|Neoplasm"
image_modality = "CT"
mask_modality = ["SEG","RTSTRUCT"]
lesion_location = "abdomen"
```

RADCURE  
* Using mit_RADCURE_windowed

```python
datasource = "TCIA"
dataset = "RADCURE"
ROI_key = "GTVp"
image_modality = "CT"
mask_modality = ["RTSTRUCT"] # THIS HAS TO BE A LIST
lesion_location = "headneck"
special_prefix = "OCSCC_"
special_suffix = "_windowed"
```