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
