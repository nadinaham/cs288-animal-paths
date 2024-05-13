#!/bin/bash

## NOTE: this is not an sbatch script, it is to generate the input to one: 'process_masks_parallel.sh'
## Processing is taking a mask tile sliced from from the mask orthomosaic by 'tile_orthomosaics.py'
## and making a separate mask for each unique path label, dilating them from 1px lines to
## fill the path

## Folder containing mask tile .npy files that were sliced from the mask orthomosaic
## by 'tile_orthomosaics.py' to process in parallel 
input_mask_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-mask-tiles/numpy-images"

## Folder to output the processed masks to, should NOT be the same as the input folder
## make sure this folder has already been created.  Make sure the final director it either ends with is 'path-masks-finished' or
## 'path-masks-finished'/'<some other directory, like masks_sep>'
## path-masks-finished should live in the same parent directory as path-mask-tiles (the parent directory of the directory of mask numpy images)
output_mask_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-masks-finished"

## Boolean: if true, separates out each unique path within the inputted mask image tile
## into a separate mask (so if there are 3 different paths labeled in the inputted image
## tile, 3 masks, each containing 1 path, will be created for this image tile).  Otherwise
## gives all the paths within the mask the same label, creating only 1 mask containing all
## path.  It is HIGHLY recommended to leave this as True, we showed experimentally that the 
## model fails otherwise.
separate_masks="True"  ## leave this as True

## Boolean: if true, dilates the path labels by 20px.  This should only be set to true when
## working with labels that are 1 px lines.  It should be set to false when using labels that
## are polygons (which enclose the paths)
dilate_masks="True"

## Absolute path to the parallel mask processing script, TODO: mk code mk this dir if does not exist
python_script="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/process_masks_parallel.py"

## Place where the scripts to process each mask (in parallel) will be stored
commands_to_run_file="\n\davies_lab\Lab\shared_projects\AnimalPathMapping\cs288-animal-paths\AnimalPathMapping\sbatchs\data_processing\mask_parallel_processing_cmds.sh"

## Make job arrray file
ls $input_mask_folder/*[0-9].npy | tr -d "*" | awk '{print "python '$python_script' "$1" '$output_mask_folder' '$separate_masks' '$dilate_masks'"}' > $commands_to_run_file


## run this using 'bash get_mask_parallel_processing_cmds.sh'