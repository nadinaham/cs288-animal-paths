#!/bin/bash

## NOTE: this is not an sbatch script, it is to generate the input to one: 'fuse_parallel_processing.sh'
## Fuses together an image tile across different image modalities.  This image tile is sliced from the
## rgb and thermal image modalities by 'tile_orthomosaics.py'

## Folder containing image tile .png files that were sliced from the orthomosaics
## by 'tile_orthomosaics.py' to process in parallel (fusing them with other image modalities,
## assumes same image has same image id across the different modalities, all end in '-id.png')
## assumes path looks like this: {image_tiles_path}/{modality}-tiles/png-images/'
## where image_tiles_path contains 'rgb-tiles' and 'thermal-tiles' folders

## input_rgb_image_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/rgb-tiles/png-images"
input_rgb_image_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rgb-tiles/png-images"

rgb_weight="0.9"

## TODO add option for including LiDAR?


## Absolute path to the parallel mask processing script
python_script="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/fuse_pngs_parallel.py"

## Place where the scripts to process each mask (in parallel) will be stored
## commands_to_run_file="\n\davies_lab\Lab\shared_projects\AnimalPathMapping\cs288-animal-paths\AnimalPathMapping\sbatchs\data_processing\mask_parallel_processing_cmds.sh"
commands_to_run_file="fuse_parallel_processing_cmds_90.sh"

## Make job arrray file
ls $input_rgb_image_folder/*[0-9].png | tr -d "*" | awk '{print "python '$python_script' "$1" '$rgb_weight'"}' > $commands_to_run_file


## run this using 'bash get_fuse_parallel_processing_cmds.sh', making sure the current fuse_parallel_processing_cmds.sh file has been deleted