'''
tile_orthomosiac by Samantha Marks and Lucia Gordon
inputs: thermal, RGB, LiDAR, and possibly labels
outputs: thermal, RGB, and LiDAR PNG images and arrays, identifiers, possibly labels, and thermal maximum pixel values
'''

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread
import sys
from sys import argv
from PIL import Image

# # global variables
# folder = argv[1]
# with_labels = True if (len(argv) > 1 and argv[2] == 'labels') else False

# # folders
# for modality in ['thermal', 'rgb', 'lidar']:
#     if not os.path.exists(f'{folder}/png-images/{modality}'):
#         os.mkdir(f'{folder}/png-images/{modality}')

# functions
def save_png_matrices_plt(raw_images, modality):
    """
    Saves numpy arrays corresponding to images as pngs by plotting them as 
    a matplotlib figure.  Also reads them back into numpy arrays and saves these.
    NOTE: this results in loss of resolution

    TODO: folder param
    """
    png_arrays = []

    for i in range(len(raw_images)):
        plt.figure(dpi = 60.7) # dpi=60.7 to get resultant arrays of (224,224,3), dpi=11 to get resultant arrays of (40,40,3), 27.3 for 100
        image = plt.imshow(raw_images[i]) # plot the array of pixel values as an image

        if modality == 'thermal' or modality == 'lidar':
            # NOTE: thermal data is type float 32, this assumes values are between and 1
            # lidar data is uint 8
            # what both thermal and lidar share in common is that they are 
            # single band (have 1 channel)
            # so cmap turns them into RGBA (for each pixel in 2D numpy array
            # replaces it with a (R, G, B, A) data value -> adds a dimension
            # to the numpy array to store this), A (trasparency) = 1, RGB
            # set based off of original pixel value
            image.set_cmap('inferno')
        
        plt.axis('off') # remove axes        
        plt.savefig(f'{folder}/png-images/{modality}/{modality}-{i}.png', bbox_inches = 'tight', pad_inches = 0) # temporarily save the image
        plt.close() # close the image to save memory
        png_arrays.append(imread(f'{folder}/png-images/{modality}/{modality}-{i}.png')) # convert the PNG image to a 3D array
    
    np.save(f'{folder}/data/{modality}/{modality}-images', png_arrays)

def save_matrices_to_png_plt2(raw_image_arrs: list, modality: str, output_folder: str):
    """
    Processes numpy arrays tiled (sliced) from a tiff of type <modality> 
    that had been processed into an orthomosaic (numpy array) by 
    'align_orthomoaics.py'.  Converts them to RGB if they weren't already
    and saves them to a png and corresponding numpy array (except for masks:
    just saves their numpy array slices as inputted).
    NOTE: this maintains the size of the tiles, in pixels.

    Parameters:
    -----------
    raw_image_arrs: list
        list of np.arrays, where each np.array is a tile (slice) of the
        orthomosaic of type <modality>

    modality: str
        type of image modality of 'raw_images' (what type of image they're
        sliced from).  
        Options: 'rgb', 'thermal', 'lidar', 'path-mask'

    output_folder: str
        absolute path to folder to store the sliced image tiles to (as 
        pngs and numpy array files).  
        
        Output directory structure will be:
        {output_folder}/{modality}-tiles/numpy-images.npy
            (contains the tiles as np arrays)
        and
        {output_folder}/{modality}-tiles/png-images/ 
            (contains the tiles as pngs)
    """
    # create folders for storing the tiles in
    if not os.path.exists(f'{output_folder}/{modality}-tiles'):
        os.mkdir(f'{output_folder}/{modality}-tiles')
    if not os.path.exists(f'{output_folder}/{modality}-tiles/png-images'):
        os.mkdir(f'{output_folder}/{modality}-tiles/png-images')
    if not os.path.exists(f'{output_folder}/{modality}-tiles/numpy-images'):
        os.mkdir(f'{output_folder}/{modality}-tiles/numpy-images')

    # will resize images to 400 by 400 to match RGB (TODO later take out hardcoding)
    # new_width = 400
    new_width = 800
    # new_width = 1200

    # arrays to hold image numpy arrays after they've been coverted to pngs
    # and reread back into numpy arrays
    png_arrays = []
    for i in range(len(raw_image_arrs)):
        if modality == "thermal" or modality == "lidar":
            # resize image to new_width (scale height accordingly)
            new_height = round(new_width * raw_image_arrs[i].shape[0] / raw_image_arrs[i].shape[1])
            # read into PIL to conserve pixel values before resizing
            im = Image.fromarray(raw_image_arrs[i])
            resize_im = im.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # preserve original np type for each modality from align_orthomosaics.py
            if modality=="thermal":
                resize_arr = np.array(resize_im).astype(np.float32)
            else:
                resize_arr = np.array(resize_im).astype('uint8')

            # since thermal and lidar images have only 1 band (1 channel),
            # need to convert them to RGB (3 channels), use inferno color
            # mapping to do so
            # NOTE: assumes thermal orthomosaic is of type np.float32, all 
            # values should be between 0 and 1
            # assumes lidar orthomosaic is of type uint.8, all values should 
            # be between 0 and 255
            # plt.imsave(f'{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png', raw_image_arrs[i], cmap="inferno")
            plt.imsave(f'{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png', resize_arr, cmap="inferno")
            # reread image saved as png back into numpy array
            png_arrays.append(imread(f'{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png'))
            

        # NOTE: maintaining RGB and mask sizes bc mask sizes are too small to resize
        # and get good performance in the Mask R-CNN model
        # TODO later take out hard coding of new size and not RGB
        elif modality == "path-mask":
            # for this modality, assumption is that for each path id 'pi' from
            # path label shapefile, all pixels belonging to path 'pi' have value
            # 'pi' at their corresponding index in the 2D path-mask numpy array
            # (orthomosaic).  So if have 1357 paths in the shapefile (and thus
            # path mask), then have values 0, 1, 2, ..., 1357 in the array (0 is
            # background).  As these values are close together, you cannot just
            # convert the path mask tiles into pngs, so they are saved as their
            # original numpy arrays for later path-mask-specific processing in
            # 'process_masks.py'
            png_arrays.append(raw_image_arrs[i])

            # Also save each numpy array image tile to its own .npy file for
            # downstream parallel processing
            np.save(f'{output_folder}/{modality}-tiles/numpy-images/{modality}-{i}.npy', raw_image_arrs[i])


        elif modality == "rgb":
             # TODO temporary fix, take out
            # if (i <= 7333):
            #     continue
            # assumes this is RGB data (has R, G, and B channels) so can
            # save as original colors
            plt.imsave(f'{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png', raw_image_arrs[i])
            # reread image in png format back into numpy array
            png_arrays.append(imread(f'{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png'))
        
        else:
            raise Exception(f"Invalid modality.  Entered: {modality}.  Modality must be one of: 'rgb', 'lidar', 'thermal', 'path-mask'")

    # save image slices read from saved pngs to list of numpy arrays
    np.save(f'{output_folder}/{modality}-tiles/numpy-images/all-numpy-images', png_arrays)

        
        
def tile_thermal_orthomosaic(orthomosaic_path: str, with_midden_labels: bool, midden_path: str, constants_path: str):
    """
    Slices the thermal orthomosaic (numpy array processed from tiff for
    thermal data by 'align_orthomosaics.py') into uniformly sized pieces.
    Optionally also includes midden labels in the slicing.

    NOTE: this must be called before tiling any of the other orthomosaics
    for any of the other image modalities.

    Parameters:
    -----------
    orthomosaic_path: str
        absolute path to file containing thermal orthomosaic (file type:
        .npy) processed by 'align_orthomosaics.py'

    midden_labels: bool
        if True: keeps track of the middens in each image tile sliced from 
        the orthomosaic

    midden_path: str
        absolute path to file containing midden orthomosaic, should be
        aligned with the thermal orthomosaic

    constants_path: str
        absolute path to file containing constants (file name starts with
        file name in 'orthomosaic_path' and ends with
        '-thermal-processing-info.npy') generated by 'align_orthomosaics.py'
        during processing of thermal tiff image into orthomosaic

    NOTE: if midden labels are used and should be processed, this tiling
    function must be called before any of the other ones for the other 
    image modalities

    Returns:
    --------
    thermal_images: list
        list containing all of the image tiles (as np arrays) sliced from
        the inputted orthomosaic

    identifiers: list
        number of image tiles sliced from the inputted orthomosaic

    max_pixel_vals: list
        list where at each index i, the maximum pixel value in image tile i
        is stored

    labels: list
        list where the value at each index i is 1 if there is a midden in
        the corresponding image tile i, 0 otherwise

    constants: list
        list containing constants from processing the thermal tiff and
        orthomosaic
    """
    # crop thermal orthomosaic
    thermal_orthomosaic = np.load(orthomosaic_path, allow_pickle=True)
    print(f"thermal orthomosaic shape: {thermal_orthomosaic.shape}")
    if with_midden_labels == True: midden_matrix = np.load(midden_path, allow_pickle=True)
    # THERMAL_INTERVAL = 40 # width and height of cropped thermal images in pixels, is 1/10 of interval for RGB for firestorm TODO
    THERMAL_INTERVAL = 80
    # THERMAL_INTERVAL = 120 # width and height of cropped thermal images in pixels, is 1/10 of interval for RGB for firestorm TODO
    THERMAL_STRIDE = 0 # overlap of cropped thermal images in pixels
    THERMAL_STEP = int(THERMAL_INTERVAL / 2 + THERMAL_STRIDE) # from before messed with all params: 30
    thermal_images = [] # images cropped from orthomosaic
    if with_midden_labels == True: thermal_label_matrices = [] # midden locations cropped from orthomosaic
    if with_midden_labels == True: thermal_midden_images = [] # subset of the cropped images that contain middens
    if with_midden_labels == True: thermal_empty_images = [] # subset of the cropped images that are empty

    for bottom in range(THERMAL_INTERVAL, thermal_orthomosaic.shape[0], THERMAL_STEP): # begin cropping from the top of the orthomosaic
        num_horizontal = 0

        for right in range(THERMAL_INTERVAL, thermal_orthomosaic.shape[1], THERMAL_STEP): # begin cropping from the left end of the orthomosaic
            cropped_image = thermal_orthomosaic[bottom - THERMAL_INTERVAL : bottom, right - THERMAL_INTERVAL : right].copy() # create an image cropped from the orthomosaic
            cropped_image -= np.amin(cropped_image) # set the minimum pixel value to 0
            thermal_images.append(cropped_image) # save cropped image to list
            if with_midden_labels == True: thermal_label_matrices.append(midden_matrix[bottom - THERMAL_INTERVAL : bottom, right - THERMAL_INTERVAL : right]) # save the same cropping from the matrix of midden locations
            num_horizontal += 1

    constants = list(np.load(constants_path, allow_pickle=True))
    constants.append(['NUM_HORIZONTAL', num_horizontal])
    if with_midden_labels == True: labels = list(np.sum(np.sum(thermal_label_matrices, axis = 1), axis = 1)) # collapses each label matrix to the number of middens in the corresponding cropped image
    identifiers = list(range(len(thermal_images)))
    max_pixel_vals = [np.amax(thermal_images[i]) for i in range(len(thermal_images))]

    if with_midden_labels == True: 
        for index in range(len(labels)):
            if labels[index] > 1: # if there happens to be more than 1 midden in an image
                labels[index] = 1 # set the label to 1 since we only care about whether there is a midden or not

    else: 
        labels = None

    return thermal_images, identifiers, max_pixel_vals, labels, constants
        

def tile_rgb_orthomosaic(orthomosaic_path: str):
    """
    Slices the RGB orthomosaic (numpy array processed from tiff for
    RGB data by 'align_orthomosaics.py') into uniformly sized pieces.

    Parameters:
    -----------
    orthomosaic_path: str
        absolute path to file containing RGB orthomosaic (file type:
        .npy) processed by 'align_orthomosaics.py'

    Returns:
    --------
    rgb_images: list
        list containing all of the image tiles (as np arrays) sliced from
        the inputted orthomosaic
    """
    # crop RGB orthomosaic
    rgb_orthomosaic = np.load(orthomosaic_path, allow_pickle=True)
    print(f"rgb orthomosaic shape: {rgb_orthomosaic.shape}")
    # RGB_INTERVAL = 400 # width of cropped thermal images in pixels, is 10x that of thermal
    RGB_INTERVAL = 800
    # RGB_INTERVAL = 1200
    RGB_STRIDE = 0 # overlap of cropped thermal images in pixels
    RGB_STEP = int(RGB_INTERVAL / 2 + RGB_STRIDE) # OLD: 300
    rgb_images = [] # images cropped from orthomosaic
    # TODO: these are unused ?
    # if with_labels == True: rgb_midden_images = [] # subset of the cropped images that contain middens
    # if with_labels == True: rgb_empty_images = [] # subset of the cropped images that are empty

    for bottom in range(RGB_INTERVAL, rgb_orthomosaic.shape[0], RGB_STEP): # begin cropping from the top of the orthomosaic
        for right in range(RGB_INTERVAL, rgb_orthomosaic.shape[1], RGB_STEP): # begin cropping from the left end of the orthomosaic
            cropped_image = rgb_orthomosaic[bottom - RGB_INTERVAL : bottom, right - RGB_INTERVAL : right].copy() # create an image cropped from the orthomosaic
            rgb_images.append(cropped_image) # save cropped image to list

    # save constants used to slice the RGB image into tiles to reuse in
    # downstream processing of animal path masks (if any)
    rgb_slicing_consts_dict = {}
    rgb_slicing_consts_dict["RGB_INTERVAL"] = RGB_INTERVAL
    rgb_slicing_consts_dict["RGB_STRIDE"] = RGB_STRIDE
    rgb_slicing_consts_dict["RGB_STEP"] = RGB_STEP

    return rgb_images, rgb_slicing_consts_dict

def tile_lidar_orthomosaic(orthomosaic_path: str):
    """
    Slices the LiDAR orthomosaic (numpy array processed from tiff for
    RGB data by 'align_orthomosaics.py') into uniformly sized pieces.

    Parameters:
    -----------
    orthomosaic_path: str
        absolute path to file containing LiDAR orthomosaic (file type:
        .npy) processed by 'align_orthomosaics.py'

    Returns:
    --------
    lidar_images: list
        list containing all of the image tiles (as np arrays) sliced from
        the inputted orthomosaic
    """
    # crop LiDAR orthomosaic
    lidar_orthomosaic = np.load(orthomosaic_path)
    print(f"lidar orthomosaic shape: {lidar_orthomosaic.shape}")
    # LIDAR_INTERVAL = 200 # is 1/2 that of RGB and 5x that of thermal
    LIDAR_INTERVAL = 400
    # LIDAR_INTERVAL = 600 # is 1/2 that of RGB and 5x that of thermal
    LIDAR_STRIDE = 0
    LIDAR_STEP = int(LIDAR_INTERVAL / 2 + LIDAR_STRIDE) # 60
    lidar_images = []

    for bottom in range(LIDAR_INTERVAL, lidar_orthomosaic.shape[0], LIDAR_STEP): # begin cropping from the top of the orthomosaic
        for right in range(LIDAR_INTERVAL, lidar_orthomosaic.shape[1], LIDAR_STEP): # begin cropping from the left end of the orthomosaic
            cropped_image = lidar_orthomosaic[bottom - LIDAR_INTERVAL : bottom, right - LIDAR_INTERVAL : right].copy() # create an image cropped from the orthomosaic
            lidar_images.append(cropped_image) # save cropped image to list

    return lidar_images

def tile_path_mask_orthomosaic(path_mask_orthomosaic_path: str, rgb_slicing_consts_dict: dict):
    """
    Slices the animal path mask orthomosaic (numpy array processed from 
    shapefile for animal path labels by 'align_orthomosaics.py') 
    into uniformly sized pieces (tiles).  Each tile corresponds to an
    RGB image tile (corresponding mask-RGB tiles will have the same index) 

    Parameters:
    -----------
    orthomosaic_path: str
        absolute path to file containing animal path mask orthomosaic 
        (file type: .npy) processed by 'align_orthomosaics.py'

    rgb_slicing_consts_dict:
        dict with the following key: value pairs:
        RGB_INTERVAL: width of RGB (and thermal) image tiles, in pixels
        RGB_STRIDE: overlap of RGB (and thermal) image tiles, in pixels
        RGB_STEP: step size for slicing RGB (and thermal) tiles, in pixels


    Returns:
    --------
    path_mask_images: list
        list containing all of the image tiles (as np arrays) sliced from
        the inputted orthomosaic
    """
    # crop animal path mask orthomosaic
    path_orthomosaic = np.load(path_mask_orthomosaic_path, allow_pickle=True)
    
    path_mask_images = [] # images cropped from orthomosaic

    # begin cropping from the top of the orthomosaic
    for bottom in range(rgb_slicing_consts_dict["RGB_INTERVAL"], path_orthomosaic.shape[0], rgb_slicing_consts_dict["RGB_STEP"]): 
        # begin cropping from the left end of the orthomosaic
        for right in range(rgb_slicing_consts_dict["RGB_INTERVAL"], path_orthomosaic.shape[1], rgb_slicing_consts_dict["RGB_STEP"]): 
            # create an image cropped from the orthomosaic
            cropped_image = path_orthomosaic[bottom - rgb_slicing_consts_dict["RGB_INTERVAL"] : bottom, right - rgb_slicing_consts_dict["RGB_INTERVAL"] : right].copy()
            path_mask_images.append(cropped_image) # save cropped image to list

    return path_mask_images


def tile_orthomosaics(thermal_orthomosaic_path: str, rgb_orthomosaic_path: str, lidar_orthomosaic_path: str, midden_orthomosaic_path: str, path_mask_orthomosaic_path: str, thermal_processing_constants_path: str, output_folder: str, save_thermal: bool, save_rgb: bool, save_lidar: bool, save_midden_mask: bool, save_path_mask: bool):
    """
    Tiles all of the orthomosaics corresponding to each type of data (thermal, 
    RGB, LiDAR, and masks).  (Slices each of the orthomosaics into uniformly
    sized image tiles).  
    
    NOTE:  the orthomosaics are numpy arrays that should be processed from 
    their corresponding tiff files by  'align_orthomosaics.py' to make sure 
    they are aligned and processed correctly).

    NOTE: if midden labels are included, they are processed through the 
    thermal orthomosaic tiling.

    Parameters:
    -----------
    thermal_orthomosaic_path: str
        absolute path to where the thermal orthomosaic to tile is stored
        NOTE: this must be included and must be a valid path to the
        thermal orthomosaic that is aligned with the other orthomosaics
        being tiled. 

    rgb_orthomosaic_path: str or None
        absolute path to where the rgb orthomosaic to tile is stored
        if None: not processing an rgb orthomosaic
    lidar_orthomosaic_path: str or None
        absolute path to where the lidar orthomosaic to tile is stored
        if None: not processing a lidar orthomosaic
    midden_orthomosaic_path: str or None
        absolute path to where the midden orthomosaic to incorporate into
        tiling of images is stored
        if None: not processing a midden orthomosaic
    path_mask_orthomosaic_path: str or None
        absolute path to where the animal path mask orthomosaic to tile is stored
        NOTE: animal path mask orthomosaic can only be tiled if RGB orthomosaic is
        NOTE: assumed to be from shapefile made overlaid on rgb tiff and 
        rasterized using same extent and resolution as rgb tiff
        if None: not processing an animal path mask orthomosaic

    thermal_processing_constants_path: str
        absolute path to file containing constants (file name starts with
        file name in 'orthomosaic_path' and ends with
        '-thermal-processing-info.npy') generated by 'align_orthomosaics.py'
        during processing of thermal tiff image into orthomosaic

    output_folder: str
        folder to store saved image tile pngs and np arrays to (see
        save_<image_modality> for details about how saved output files 
        will be structured within this directory).

    save_<image_modality>: bool
        if False: not saving the tiled orthomosaic images for the 
        <image_modality> data to pngs and their corresponding numpy arrays
        to a .npy file

        otherwise: each image tile i will be saved as a png at the location
         '{output_folder}/{modality}-tiles/png-images/{modality}-{i}.png'
        and its corresponding numpy array will be stored at index i of the
        list stored at the location
         '{output_folder}/{modality}-tiles/numpy-images.npy'  
    """
    # tile orthomosaics

    # Check to see if tiling with midden labels 
    with_midden_labels = False
    if midden_orthomosaic_path is not None:
        with_midden_labels = True

    # First must tile thermal orthomosaic
    thermal_images, identifiers, max_pixel_vals, labels, constants = tile_thermal_orthomosaic(thermal_orthomosaic_path, with_midden_labels, midden_orthomosaic_path, thermal_processing_constants_path)
    
    # Then tile any other orthomosaic specified to be tiled
    if (rgb_orthomosaic_path is not None):
        rgb_images, rgb_slicing_consts_dict = tile_rgb_orthomosaic(rgb_orthomosaic_path)

    if (lidar_orthomosaic_path is not None):
        lidar_images = tile_lidar_orthomosaic(lidar_orthomosaic_path)
    
    # must have sliced the RGB orthomosaic to slice the animal path mask orthomosaic
    if (path_mask_orthomosaic_path is not None and rgb_orthomosaic_path is not None):
        path_masks = tile_path_mask_orthomosaic(path_mask_orthomosaic_path, rgb_slicing_consts_dict)

    print("Number of images of each modality before removing empty images:")
    print(len(thermal_images), ' thermal images')
    if rgb_orthomosaic_path is not None:
        print(len(rgb_images), ' RGB images')
    if lidar_orthomosaic_path is not None:
        print(len(lidar_images), ' LiDAR images')
    if path_mask_orthomosaic_path is not None:
        print(len(path_masks), ' animal path mask images')

    # remove images that are all-black in RGB and their corresponding masks
    # TODO just merge with remove empty images below once fix identifiers issue
    # where have a diff num of pictures for diff image modalities
    # TODO: way of checking if < 1/10 of image non-black?
    # saved_indices = []
    # if rgb_orthomosaic_path is not None:
    #     for i in reversed(range(len(rgb_images))):
    #         # skip all-black RGB image and its corresponding mask
    #         if np.all(rgb_images[i] == 0):
    #             del rgb_images[i]
    #             if path_mask_orthomosaic_path is not None:
    #                 del path_masks[i]
    #             # TODO this will fail if has a diff number of images
    #             # if lidar_orthomosaic_path is not None:
    #             #     del lidar_images[i]
    #         # image tile i saved successfully, keep track of this
    #         else:
    #             saved_indices.append(i)


    # remove empty images
    for i in reversed(range(len(identifiers))): # NOTE: num of identifiers is not same across all modalities currently
        # if an image is all black in any of the modalities being tiled (out 
        # of thermal, RGB, or LiDAR), remove it in all modalities
        # exclude path masks from if statement because it is valid to have 
        # an empty mask if there are no paths in the image
        # TODO should this be if its all black in any modality? (is this fine in lidar?)
        # (or remove if its all black in rgb?)
        # if (np.all(thermal_images[i] == 0) and (rgb_orthomosaic_path is None or (rgb_orthomosaic_path is not None and np.all(rgb_images[i] == 0)))
        #     and (lidar_orthomosaic_path is None or (lidar_orthomosaic_path is not None and np.all(lidar_images[i] == 0)))):
        if (np.all(thermal_images[i] == 0) or (rgb_orthomosaic_path is not None and np.all(rgb_images[i] == 0))
            or (lidar_orthomosaic_path is not None and np.all(lidar_images[i] == 0))):
            del thermal_images[i]
            if rgb_orthomosaic_path is not None:
                del rgb_images[i]
            if lidar_orthomosaic_path is not None:
                del lidar_images[i]
            # Need to also get rid of path mask corresponding to a deleted image
            if path_mask_orthomosaic_path is not None:
                del path_masks[i]
            if with_midden_labels: 
                del labels[i]
            del max_pixel_vals[i]
            del identifiers[i]

    print("Number of image tiles for each modality tiled:")
    print(len(thermal_images), ' thermal images')
    if rgb_orthomosaic_path is not None:
        print(len(rgb_images), ' RGB images')
    if lidar_orthomosaic_path is not None:
        print(len(lidar_images), ' LiDAR images')
    if path_mask_orthomosaic_path is not None:
        print(len(path_masks), ' animal path mask images')
    print("Additional statistics:")
    print(len(max_pixel_vals), ' maximum pixel values')
    # would need to loop through list
    # print(identifiers, ' identifiers list (list of the indices of the image tiles that were saved, corresponds to all image modalities)')
    # TODO put back in once all modalities are tiled into same number of images, replace savied_indices
    print(len(identifiers), ' number of unique image identifiers saved')
    # print(len(saved_indices), ' number of unique RGB image tiles and corresponding masks saved')
    if with_midden_labels: 
        print(len(labels), ' midden labels')
    if with_midden_labels: 
        print(np.sum(labels), ' midden images')

    # save data
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if save_thermal:
        save_matrices_to_png_plt2(thermal_images, 'thermal', output_folder)
        print('thermal images saved')

    if save_rgb:
        save_matrices_to_png_plt2(rgb_images, 'rgb', output_folder)
        print('RGB images saved')
    
    if save_path_mask:
        save_matrices_to_png_plt2(path_masks, "path-mask", output_folder)

    if save_lidar:
        save_matrices_to_png_plt2(lidar_images, 'lidar', output_folder)
        print('LiDAR images saved')

    # TODO in future mkoutput dir if doesn't exist

    np.save(f'{output_folder}/tile-processing-constants', constants)
    print('constants saved')

    if with_midden_labels == True: np.save(f'{output_folder}/midden-labels', labels)
    if with_midden_labels == True: np.save(f'{output_folder}/midden-label-indices', list(range(len(labels))))
    if with_midden_labels == True: print('labels saved')

    # TODO put back in once all modalities are tiled into same number of images, replace saved_indices
    np.save(f'{output_folder}/image-tile-identifiers', identifiers)
    print('identifiers saved')
    # np.save(f'{output_folder}/image-tile-ids-saved', saved_indices)
    # print('indices of saved images saved')

    np.save(f'{output_folder}/max-pixel-vals', max_pixel_vals)
    print('max pixel vals saved')


# TODO remove this and put in separate scripts folder outside innermost AnimalPathMapping directory
if __name__ == '__main__':
    # process system arguments (excluding script name)
    arguments = [None] * (len(sys.argv) - 1)
    print("System arguments:")
    for i in range(len(sys.argv) - 1):
        # idx 1 of sys.argv is python script name, skip that
        print(sys.argv[i+1])
        # process None, False, True into Python None object, bool False, 
        # bool True, respectively
        if sys.argv[i+1] == "None":
            arguments[i] = None
        elif sys.argv[i+1] == "True":
            arguments[i] = True
        elif sys.argv[i+1] == "False":
            arguments[i] = False
        else:
            arguments[i] = sys.argv[i+1]

    
    
    # process each of the tiffs passed in into numpy arrays, aligning them
    # with each other, and saving them to the specified file path (if saving)
    tile_orthomosaics(*arguments)


# TODO instead of hardcoding interval, calculate it
# RGB is 10* thermal, LIDAR is 5*thermal
# aka factor of diff btn shapes (for height and width sep, should be same)

# TODO: after processing masks into dilated pngs, need to resize everything so same size
# maybe should just save the rgbs, thermal, and lidar images to the desired same size here
# right now they're all getting saved to diff sizes, Lucia's dpi thing was to set them to
# same size
# Lucia also recommending slicing over a wider area than 400x400px bc paths are long and
# then just downscaling to smaller size