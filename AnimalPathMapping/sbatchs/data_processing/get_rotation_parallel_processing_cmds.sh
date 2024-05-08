#!/bin/bash

## NOTE: this is not an sbatch script, it is to generate the input to one: 'rotate_tiles.sh'
## This takes an RGB image and its corresponding mask and rotates them 8 times, by 45 degrees
## each time, and then saves them in the format required for model training

## Assumes that masks were processed by 'process_masks_parallel.py' and are stored in
## a folder called 'path-masks-finished', whose parent directory has another child directory
## with the images (of the modality in use), which is called '{modality}-tiles'
## So assumes the path stucture should look like this:{image_tiles_path}/{modality}-tiles/png-images/'
##                                                                      /path-masks-finished/{image_{ident}_segmentation_masks', which contains .npy files
## where '{image_tiles_path}/rgb-tiles/png-images/' contains image files of the form '{modality}-{ident}.png'
## and ident matches up to ident inside '/path-masks-finished/*/{image_{ident}_segmentation_masks}'
## (so this folder contains the masks corresponding to image with id {ident}
input_image_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rgb-tiles/png-images"

## if "True", also rotates the masks with the same image id (these masks correspond to the inputted image)
## otherwise, set to "False"
rotate_corresp_masks="True"


## assumes that masks for image with identity {ident} are stored along the path:
## '/path-masks-finished/{*}/image_{ident}_segmentation_masks'
## specify all values that * should take on by listing them in the array below
## (note that they must be separated by spaces).  Note: if the desired masks to
## rotate are not stored in an intermediary folder * between 'path-masks-finiished' 
## and 'image_{ident}_segmentation_masks', include "None" in the list.  For example:
## mask_folders = (None, masks_sep)
## would result in rotating all masks inside '/path-masks-finished/image_{ident}_segmentation_masks'
## as well as all masks inside '/path-masks-finished/masks_sep/image_{ident}_segmentation_masks'
mask_folders=(
    masks_sep masks_tog_fixed
)

mask_folders_str=$(IFS=,; echo "${mask_folders[*]}")

## Absolute path to the parallel rotation script
python_script="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/rotate_parallel.py"

## Place where the scripts to process each mask (in parallel) will be stored
## commands_to_run_file="\n\davies_lab\Lab\shared_projects\AnimalPathMapping\cs288-animal-paths\AnimalPathMapping\sbatchs\data_processing\mask_parallel_processing_cmds.sh"
commands_to_run_file="rotation_parallel_processing_cmds_both.sh"

## Make job arrray file
# ls $input_image_folder/*[0-9].png | tr -d "*" | awk '{print "python '$python_script' "$1" '$rotate_corresp_masks'"}' > $commands_to_run_file
ls $input_image_folder/*[0-9].png | tr -d "*" | awk '{print "python '$python_script' "$1" '$rotate_corresp_masks' '$mask_folders_str'"}' > $commands_to_run_file
## Note that NR will be the new image id (NR is the current awk record number)
# ls $input_image_folder/*[0-9].png | tr -d "*" | awk '{print "python '$python_script' "$1" '$rotate_corresp_masks' '$mask_folders_str' "NR"'"}' > $commands_to_run_file

## run this using 'bash get_rotation_parallel_processing_cmds.sh'