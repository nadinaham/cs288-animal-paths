"""
process_masks_parallel by Samantha Marks
inputs: tiled animal path mask images (as .npy files) outputted from
        'tile_orthomosaic.py' using the path mask options (must have provided
        a path to the animal path mask orthomosaic AND set save for the path
        masks to True)
outputs: processed mask tiles, such that each path mask tile corresponding to
        image tile i (of the RGB/thermal tiles) gets a new directory which
        contains a set of masks (as .npy for model training and .pngs visualization)
        for each path with a unique label within the tile.  (See 'process_mask()' for
        details).  This ensures that each uniquely labeled path within a tiled 
        image gets its own individual mask, with one mask per distinct path 
        in each tiled image.  

To run, use 'process_masks_parallel.sh' under 'AnimalPathMapping/sbatchs/data_processing'
and fill in the variables appropriately (read all instructions carefully, it will refer you
to using 'get_mask_parallel_processing_cmds.sh')
"""

import os
import numpy as np
import sys
import cv2 as cv
from PIL import Image


def process_mask(mask_path: str, output_path: str, separate_masks: bool, dilate: bool):
    """
    Processes a mask file stored as a 2D numpy array of type int16 possibly
    containing several path labels, where each unique label corresponds to a
    unique "pixel" value (a unique integer value in the array). 

    Stores the masks using the following file directory structure, supposing 
    the .npy file name at the end of 'mask_path' is 'mask-path-{i}.npy' and
    that this mask file has 2 unique path labels:
        {output_path}/image_{i}_segmentation_masks/
            image_{i}_seg_mask_0.npy
            image_{i}_seg_mask_1.npy
        {output_path}/image_{i}_visualization_masks/
            image_{i}_vis_mask_0.png
            image_{i}_vis_mask_1.png
     
    Parameters:
    -----------
    mask_path: str
        absolute path to mask tile .npy file (tile created by 'tile_orthomosaic.py')
        to be processed into masks that an image segmentation model can use.

    output_path: str
        absolute path to folder to output the processed masks to. NOTE: for downstream
        processing, this should end with 'path-masks-finished/' where the parent directory
        of the 'path-masks-finished' directory is same as the parent directory for 'path-mask-tiles'
        (created by 'tile_orthomosaic.py'), where the 'mask_path' is a file stored in 'path-mask-tiles'

    separate_masks: bool
        if True: creates a separate mask file for each unique path label in 
        the inputted mask file, stored as a type np.uint8 numpy array (for 
        use in passing to a segmentation model), as well as a .png file 
        (solely for purposes of visualization).

        otherwise: creates 1 label for all of the distinct path labels
            within the mask tile (so all paths within the image tile will 
            share the same label)

    dilate: bool
        if True, expands the mask labels by 20px (this is for masks that are
            1 px lines, not ones labeled using polygons)
        otherwise: processes the masks as labeled, does not dilate them or
            modify their size in any way
    """
    if not os.path.exists(output_path):
        try:
            os.mkdir(output_path)
        except:
            # assume some other process running in parallel made this directory
            # (Note: may also fail if none of the directories above the current directory
            # exist, this is due to our suggestion that this output_path ends in a directory
            # which is a child directory of an already existing parent directory containing the
            # directory from which the mask being processed is stored)
            pass

    # TODO make output_path directory using os.makedirs to get the full file path (code currently
    # breaks if outermost output_path dir not already amde)
    # Get base name of mask_path (just the file_name.extension at the end
    # of the path)
    base_name = os.path.basename(mask_path)
    file_name, extension = os.path.splitext(base_name)
    # verify is a .npy file
    if (extension != ".npy"):
        raise Exception(f"Invalid file type entered, your type was {extension}, it should have been .npy.  See the calling function's docstring for what 'mask_path' should be.")
    # get id of image mask corresponds to (by the mask's id, which should
    # be the same as its corresponding image's id), this assumes file name
    # ends with "-id"
    ident = file_name.split("-")[-1]
    # verify is a number so that it can be a valid id
    if not ident.isdigit():
        raise Exception(f"Invalid file name entered, ends with: {ident}.  Must end with the id number (as an integer) of the mask, which is the same as the image id the inputted mask corresponds to.")


    if separate_masks:
        output_path = f'{output_path}/masks_sep'
    else:
        output_path = f'{output_path}/masks_tog_fixed'
    if dilate:
        output_path = f'{output_path}_dilated'
    
    # Create file directory structure for processed mask
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except:
            # Assumes some other processing running parallel made the output path
            pass

    seg_mask_folder = f'{output_path}/image_{ident}_segmentation_masks'
    vis_mask_folder = f'{output_path}/image_{ident}_visualization_masksv'

    if not os.path.exists(seg_mask_folder):
        os.mkdir(seg_mask_folder)
    if not os.path.exists(vis_mask_folder):
        os.mkdir(vis_mask_folder)
    mask_seg_basename = f'{seg_mask_folder}/image_{ident}_seg_mask'
    mask_vis_basename = f'{vis_mask_folder}/image_{ident}_vis_mask'

    # Load mask numpy array
    mask = np.load(mask_path, allow_pickle=True)


    # Create kernel for dilating 1px path mask label to being 20px wide
    # (this is chosen based off of visual examination of the animal path widths
    # to ensure the dilated labels will cover the full width of the paths they label)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))

    # for converting to png: value 0 in 2D binary array maps to (0, 0, 0)
    # and value 1 in 2D binary array maps to (255, 255, 255)
    # (values in 2D binary array get used as indices)
    lookup = np.array([[0,0,0], [255,255,255]]).astype(np.uint8)

    # get number of unique paths (by number of unique path labels, where each
    # label is encoded as a distinct pixel color)
    unique_vals = np.unique(mask)
        
    if len(unique_vals) == 1:
        # only color in mask is black (=0) (gauranteed bc path will not 
        # fill entire image), make an all-black mask
        # save mask array in form for segmentation model and form for visualization
        mask = mask.astype(np.uint8) # model requires masks to be of type np.uint8
        np.save(f'{mask_seg_basename}_0.npy', mask)

        # Convert data to 3D array using (0, 0, 0) for value 0 and (255, 255, 255)
        # for value 1 in order to save it as a png
        vis_mask = lookup[mask]
        vis_mask_img = Image.fromarray(vis_mask, 'RGB')
        vis_mask_img.save(f'{mask_vis_basename}_0.png')
        return 

    if separate_masks:
        # For each unique value in the array, make a separate mask file
        # containing just pixels with that value

        # skip 0: background (guar to be 0 bc 0 in mask and path won't cover whole image
        # and mask is black with 0 as lowest val)
        for i in range(1, len(unique_vals)): 
            # make a new mask as a 2D binary array, where all pixels of value i
            # are set to 1, and all other pixels of original mask are set to 0 (black)
            # note:  model requires masks to be saved as type np.uint8
            unique_path_mask = np.where(mask==unique_vals[i], 1 , 0).astype(np.uint8)
            if dilate:
                # dilate path label to be 20px wide from being 1px wide
                unique_path_mask = cv.dilate(unique_path_mask, kernel).astype(np.uint8)
                # save mask array in form for segmentation model and form for visualization
            np.save(f'{mask_seg_basename}_{i}.npy', unique_path_mask)

            # Convert data to 3D array using (0, 0, 0) for value 0 and (255, 255, 255)
            # for value 1 in order to save it as a png
            vis_mask = lookup[unique_path_mask]
            vis_mask_img = Image.fromarray(vis_mask, 'RGB')
            vis_mask_img.save(f'{mask_vis_basename}_{i}.png')

    else:
        # all masks get the same label (even if they had unique ids in the input),
        # include all paths in one mask, noting that 0 is background, so >0 is path
        mask[mask > 0] = 1
        # Need to cast to uint8 to be able to feed into the model
        mask = mask.astype(np.uint8)
        if dilate:
            mask = cv.dilate(mask, kernel).astype(np.uint8)
        np.save(f'{mask_seg_basename}_1.npy', mask)

        # Convert data to 3D arrayf using (0, 0, 0) for value 0 and (255, 255, 255)
        # for value 1 in order to save it as a png
        vis_mask = lookup[mask]
        vis_mask_img = Image.fromarray(vis_mask, 'RGB')
        vis_mask_img.save(f'{mask_vis_basename}_1.png')

        
if __name__ == '__main__':
    print("System arguments:")
    for i in range(len(sys.argv) - 1):
        # idx 1 of sys.argv is python script name, skip that
        print(sys.argv[i+1])

    # check if processing the mask to make a unique mask for each unique path label
    if (sys.argv[3] == "True"):
        sep_masks = True
    else:
        sep_masks = False
    
    if (sys.argv[4] == "True"):
        dilate_masks = True
    else:
        dilate_masks = False

    process_mask(sys.argv[1], sys.argv[2], sep_masks, dilate_masks)




