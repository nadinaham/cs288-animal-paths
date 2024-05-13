'''
align-orthomosaics by Samantha Marks, Lucia Gordon, and Samuel Collier
inputs: thermal, RGB, LiDAR, and mask tiffs
outputs: thermal, RGB, LiDAR, and mask matrices (numpy arrays)

To run, use 'process_orthomosaics.sh' under 'AnimalPathMapping/sbatchs/data_processing'
and fill in the variables appropriately.
OR: type in the terminal:
python align_orthomosaics.py thermal_tiff_path, rgb_tiff_path, lidar_tiff_path, 
    mask_tiff_path, save_thermal, save_rgb, save_lidar, save_mask

NOTE: see 'align_process_tiffs()' function below for specifications of the
arguments listed above.
'''

# imports
import os
import numpy as np
import sys
from osgeo import gdal
from sys import argv

# functions
def crop_array(matrix):
    start_row = 0
    end_row = matrix.shape[0]
    start_col = 0
    end_col = matrix.shape[1]

    # finds the first row with data
    for row_index in range(len(matrix)):
        if any(matrix[row_index] != 0):
            start_row = row_index
            break

    # truncate image to start at first row with data
    matrix = matrix[start_row:]

    # finds the last row with data
    for row_index in range(len(matrix)): 
        if all(matrix[row_index] == 0):
            end_row = row_index
            break
        else:
            end_row = matrix.shape[0]

    # truncate image to end at last row with data
    matrix = matrix[:end_row]
    
    # find the first column with data
    for col_index in range(len(matrix.T)):
        if any(matrix.T[col_index] != 0):
            start_col = col_index
            break

    # truncate the image to start at first column with data
    matrix = matrix.T[start_col:].T

    # find last column with data
    for col_index in range(len(matrix.T)):
        if all(matrix.T[col_index] == 0):
            end_col = col_index
            break
        else:
            end_col = matrix.shape[1]

    # truncate image to end at last column with data
    matrix = matrix.T[:end_col].T

    # return the indices of the first row, last row, first column, and last column
    # from the original image array that the image got truncated to 
    return start_row, start_row + end_row, start_col, start_col + end_col

def process_thermal_orthomosaic(thermal_tiff_path: str, save_image: str):
    """
    Processes thermal image, normalizing pixels (sets minimum to 0 and
    scales all other pixel values accordingly), and cropping to just the
    region of the tiff containing data (non-zero pixels).  Saves the
    processed thermal image as a numpy array.

    NOTE: thermal image must be the first to be processed in order to align
    the rest of the images (the coordinates from where it is cropped from the 
    original image are used to crop the other images. This assumes all the images
    (thermal, RGB, and LiDAR have the same dimension and were originally aligned)

    NOTE: also saves important constants for downstream processing of 
    orthomosaics to a file called {save_image}-thermal-processing-constants.npy

    Parameters:
    -----------
    thermal_tiff_path: str
        absolute path to where the thermal tiff to be processed is stored
    save_image: str or None
        if not None: the processed mask image is saved as a numpy array to
        the file named specified by save_image
        otherwise: processed mask image not saved

    Returns:
    --------
    cropping_coordinates: dict
        keys: THERMAL_TOP_FINAL, THERMAL_LEFT_FINAL, THERMAL_BOTTOM_FINAL, 
        THERMAL_RIGHT_FINAL

        values: coordinate (in meters, the unit the images were taken in) 
            corresponding to the original thermal image that the processed 
            image is cropped from.
            key and its corresponding value:
            TOP: the coordinate from the original thermal image where the 
                top edge of the cropped thermal image is
            LEFT: the coordinate from the original thermal image where the 
                left-most edge of the cropped thermal image is
            BOTTOM: the coordinate from the original thermal image where the 
                bottom edge of the cropped thermal image is
            RIGHT: the coordinate from the original thermal image where the 
                right-most edge of the cropped thermal image is

    thermal_orthomosaic_shape: 2D np.array
        array containing the shape of the processed thermal orthomosaic np array

    thermal_interval: int
        width of cropped thermal image in pixels (?)
    """
    # process thermal orthomosaic
    print('thermal orthomosaic')
    THERMAL_INTERVAL = 400 # width of cropped thermal images in pixels
    THERMAL_STRIDE = 100 # overlap of cropped thermal images in pixels

    thermal_dataset = gdal.Open(thermal_tiff_path) # converts the tiff to a Dataset object

    # extract information from the thermal tiff for cropping
    THERMAL_NUM_ROWS = thermal_dataset.RasterYSize # pixels
    THERMAL_NUM_COLS = thermal_dataset.RasterXSize # pixels

    THERMAL_PIXEL_HEIGHT = thermal_dataset.GetGeoTransform()[5] # m
    THERMAL_PIXEL_WIDTH = thermal_dataset.GetGeoTransform()[1] # m

    THERMAL_TOP = thermal_dataset.GetGeoTransform()[3] # m
    THERMAL_LEFT = thermal_dataset.GetGeoTransform()[0] # m
    THERMAL_BOTTOM = THERMAL_TOP + THERMAL_PIXEL_HEIGHT * THERMAL_NUM_ROWS # m
    THERMAL_RIGHT = THERMAL_LEFT + THERMAL_PIXEL_WIDTH * THERMAL_NUM_COLS # m
    print('top = ' + str(THERMAL_TOP) + ', bottom = ' + str(THERMAL_BOTTOM) + ', left = ' + str(THERMAL_LEFT) + ', right = ' + str(THERMAL_RIGHT))

    thermal_band = ((thermal_dataset.GetRasterBand(4)).ReadAsArray(0, 0, THERMAL_NUM_COLS, THERMAL_NUM_ROWS).astype(np.float32)) # 4th band corresponds to thermal data
    THERMAL_ORTHOMOSAIC_MIN = np.amin(np.ma.masked_less(thermal_band, 2000)) # min pixel value in orthomosaic, excluding background
    thermal_orthomosaic = np.ma.masked_less(thermal_band - THERMAL_ORTHOMOSAIC_MIN, 0).filled(0) # downshift the pixel values such that the min of the orthomosaic is 0 and set the backgnp.around pixels to 0
    print('original orthomosaic shape =', thermal_orthomosaic.shape) # pixels

    THERMAL_START_ROW, THERMAL_END_ROW, THERMAL_START_COL, THERMAL_END_COL = crop_array(thermal_orthomosaic) # extract indices for cropping
    print('start row = ' + str(THERMAL_START_ROW) + ', end row = ' + str(THERMAL_END_ROW) + ', start col = ' + str(THERMAL_START_COL) + ', end col = ' + str(THERMAL_END_COL))

    thermal_orthomosaic = thermal_orthomosaic[THERMAL_START_ROW : THERMAL_END_ROW, THERMAL_START_COL : THERMAL_END_COL] # crop out rows and columns that are 0
    print('orthomosaic shape after cropping =', thermal_orthomosaic.shape) # pixels

    new_thermal_rows = np.zeros((int(np.ceil((thermal_orthomosaic.shape[0] - THERMAL_INTERVAL) / (THERMAL_INTERVAL / 2 + THERMAL_STRIDE))) * int(THERMAL_INTERVAL / 2 + THERMAL_STRIDE) + THERMAL_INTERVAL - thermal_orthomosaic.shape[0], thermal_orthomosaic.shape[1])) # add rows so that nothing gets cut off in cropping
    thermal_orthomosaic = np.vstack((thermal_orthomosaic, new_thermal_rows)) # add rows to bottom of thermal orthomosaic
    new_thermal_cols = np.zeros((thermal_orthomosaic.shape[0], int(np.ceil((thermal_orthomosaic.shape[1] - THERMAL_INTERVAL) / (THERMAL_INTERVAL / 2 + THERMAL_STRIDE))) * int(THERMAL_INTERVAL / 2+THERMAL_STRIDE) + THERMAL_INTERVAL - thermal_orthomosaic.shape[1])) # add columns so that nothing gets cut off in cropping
    thermal_orthomosaic = np.hstack((thermal_orthomosaic, new_thermal_cols)) # add columns to right of thermal orthomosaic
    print('orthomosaic shape after adjusting size for future cropping =', thermal_orthomosaic.shape) # pixels

    # calculate the new coordinates (in meters) of the edges of the cropped thermal image
    THERMAL_TOP_FINAL = THERMAL_TOP + THERMAL_START_ROW * THERMAL_PIXEL_HEIGHT # m
    THERMAL_LEFT_FINAL = THERMAL_LEFT + THERMAL_START_COL * THERMAL_PIXEL_WIDTH # m
    THERMAL_BOTTOM_FINAL = THERMAL_TOP_FINAL + THERMAL_PIXEL_HEIGHT * thermal_orthomosaic.shape[0] # m
    THERMAL_RIGHT_FINAL = THERMAL_LEFT_FINAL + THERMAL_PIXEL_WIDTH * thermal_orthomosaic.shape[1] # m

    print('final top = ' + str(THERMAL_TOP_FINAL) + ', final bottom = ' + str(THERMAL_BOTTOM_FINAL) + ', final left = ' + str(THERMAL_LEFT_FINAL) + ', final right = ' + str(THERMAL_RIGHT_FINAL))

    constants = [['THERMAL_INTERVAL', THERMAL_INTERVAL], ['THERMAL_STRIDE', THERMAL_STRIDE], ['THERMAL_LEFT_FINAL', THERMAL_LEFT_FINAL], ['THERMAL_TOP_FINAL', THERMAL_TOP_FINAL], ['THERMAL_PIXEL_WIDTH', THERMAL_PIXEL_WIDTH], ['THERMAL_PIXEL_HEIGHT', THERMAL_PIXEL_HEIGHT], ['THERMAL_ORTHOMOSAIC_ROWS', thermal_orthomosaic.shape[0]], ['THERMAL_ORTHOMOSAIC_COLS', thermal_orthomosaic.shape[1]]]

    if save_image is not None:
        # np.save(f'{folder}/data/constants', constants)
        filename, file_extension = os.path.splitext(save_image)
        constants_file_name = filename + "-thermal_processing_info"
        np.save(f'{constants_file_name}', constants)
        # np.save(f'{folder}/data/thermal/thermal-orthomosaic-matrix', thermal_orthomosaic) # save thermal orthomosaic as numpy array
        np.save(f'{save_image}', thermal_orthomosaic) # save thermal orthomosaic as numpy array
    
    # aggregate together the values to return
    cropping_coordinates = {}
    cropping_coordinates["THERMAL_TOP_FINAL"] = THERMAL_TOP_FINAL
    cropping_coordinates["THERMAL_LEFT_FINAL"] = THERMAL_LEFT_FINAL
    cropping_coordinates["THERMAL_BOTTOM_FINAL"] = THERMAL_BOTTOM_FINAL
    cropping_coordinates["THERMAL_RIGHT_FINAL"] = THERMAL_RIGHT_FINAL
    print(f"thermal cropping coordinates: {cropping_coordinates}")

    thermal_orthomosaic_shape = thermal_orthomosaic.shape
    return cropping_coordinates, thermal_orthomosaic_shape, THERMAL_INTERVAL

def process_RGB_orthomosaic_aligned_with_thermal(rgb_tiff_path: str, thermal_cropping_coordinates_m: dict, save_image: str):
    """
    Processes RGB image, cropping to just the region of the thermal tiff 
    containing data (non-zero pixels)--this aligns the RGB orthomosaic 
    with the thermal orthomosaic previously processed.  
    Saves the processed RGB image as a uint8 numpy array if save_image.

    NOTE: thermal image must be the first to be processed in order to align
    the rest of the images (the coordinates from where it is cropped from the 
    original image are used to crop the other images. This assumes all the images
    (thermal, RGB, and LiDAR) were originally aligned (covered the same region
    in the same orientation)

    Parameters:
    -----------
    rgb_tiff_path: str
        absolute path to where rgb tiff to be processed is stored

    cropping_coordinates_m: dict
        keys: THERMAL_TOP_FINAL, THERMAL_LEFT_FINAL, THERMAL_BOTTOM_FINAL, 
        THERMAL_RIGHT_FINAL

        values: coordinates in *meters* corresponding to the original thermal
            image that the processed image is cropped from.
            key and its corresponding value:
            TOP: the coordinate from the original thermal image where the 
                top edge of the cropped thermal image is
            LEFT: the coordinate from the original thermal image where the 
                left-most edge of the cropped thermal image is
            BOTTOM: the coordinate from the original thermal image where the 
                bottom edge of the cropped thermal image is
            RIGHT: the coordinate from the original thermal image where the 
                right-most edge of the cropped thermal image is
        NOTE: this is outputtd by process_thermal_orthomosaic

    save_image: bool
        if True: the processed thermal image is saved as a numpy array to
        the file named specified by save_image
        otherwise: processed thermal image is not saved

    Returns:
    --------
    rgb_cropping_coordinates_px: dict
        key: value pairs
        "top": coordinate (in *pixels*) where the top edge of the cropped RGB
            image is relative to the original RGB image
        "bottom": coordinate (in *pixels*) where the bottom edge of the cropped RGB
            image is relative to the original RGB image
        "left": coordinate (in *pixels*) where the left edge of the cropped RGB
            image is relative to the original RGB image
        "right": coordinate (in *pixels*) where the right edge of the cropped RGB
            image is relative to the original RGB image
        NOTE: these can be used to crop a mask tiff rasterized (from a 
        shapefile aligned with the RGB tiff image) using the extent and 
        resolution  of the RGB image to align it with the RGB image
    """
    # process RGB orthomosaic
    print('RGB orthomosaic')

    rgb_dataset = gdal.Open(rgb_tiff_path) # converts the tiff to a Dataset object

    RGB_NUM_ROWS = rgb_dataset.RasterYSize # pixels
    RGB_NUM_COLS = rgb_dataset.RasterXSize # pixels
    RGB_NUM_BANDS = rgb_dataset.RasterCount # 3 bands

    RGB_PIXEL_HEIGHT = rgb_dataset.GetGeoTransform()[5] # m
    RGB_PIXEL_WIDTH = rgb_dataset.GetGeoTransform()[1] # m
    print(f"rgb pixel height: {RGB_PIXEL_HEIGHT}, width: {RGB_PIXEL_WIDTH}")

    RGB_TOP = rgb_dataset.GetGeoTransform()[3] # m
    RGB_LEFT = rgb_dataset.GetGeoTransform()[0] # m
    RGB_BOTTOM = RGB_TOP + RGB_PIXEL_HEIGHT * RGB_NUM_ROWS # m
    RGB_RIGHT = RGB_LEFT + RGB_PIXEL_WIDTH * RGB_NUM_COLS # m
    print('RGB top = ' + str(RGB_TOP) + ', bottom = ' + str(RGB_BOTTOM) + ', left = ' + str(RGB_LEFT) + ', right = ' + str(RGB_RIGHT))

    rgb_bands = np.zeros((RGB_NUM_ROWS, RGB_NUM_COLS, RGB_NUM_BANDS)) # empty RGB orthomosaic

    for band in range(RGB_NUM_BANDS):
        rgb_bands[:,:,band] = (rgb_dataset.GetRasterBand(band+1)).ReadAsArray(0, 0, RGB_NUM_COLS, RGB_NUM_ROWS) # add band data to RGB orthomosaic

    print('original orthomosaic shape =', rgb_bands.shape)


    THERMAL_TOP_FINAL = thermal_cropping_coordinates_m["THERMAL_TOP_FINAL"]
    THERMAL_LEFT_FINAL = thermal_cropping_coordinates_m["THERMAL_LEFT_FINAL"]
    THERMAL_BOTTOM_FINAL = thermal_cropping_coordinates_m["THERMAL_BOTTOM_FINAL"]
    THERMAL_RIGHT_FINAL = thermal_cropping_coordinates_m["THERMAL_RIGHT_FINAL"]
    print(f"thermal cropping coordinates: {thermal_cropping_coordinates_m}")
    # get the coordinates (in pixels) to crop the RGB orthomosaic at to cover 
    # the same area as the thermal orthomosaic (by matching up the meters coords
    # of both the thermal and RGB orthomosaics and then converting to RGB pixel coords)
    rgb_cropping_coordinates_px = {}
    rgb_cropping_coordinates_px["top"] = int((THERMAL_TOP_FINAL - RGB_TOP) / RGB_PIXEL_HEIGHT)
    rgb_cropping_coordinates_px["bottom"] = int(rgb_bands.shape[0] + (THERMAL_BOTTOM_FINAL - RGB_BOTTOM) / RGB_PIXEL_HEIGHT)
    rgb_cropping_coordinates_px["left"] = int((THERMAL_LEFT_FINAL - RGB_LEFT) / RGB_PIXEL_WIDTH) 
    rgb_cropping_coordinates_px["right"] = int(rgb_bands.shape[1] + (THERMAL_RIGHT_FINAL - RGB_RIGHT) / RGB_PIXEL_WIDTH)
    print(f"rgb cropping coordinates, px: {rgb_cropping_coordinates_px}")

    # calculate the new coordinates (in meters) of the edges of the cropped thermal image
    RGB_TOP_FINAL = RGB_TOP + rgb_cropping_coordinates_px["top"] * RGB_PIXEL_HEIGHT # m
    RGB_LEFT_FINAL = RGB_LEFT + rgb_cropping_coordinates_px["left"] * RGB_PIXEL_WIDTH # m
    RGB_BOTTOM_FINAL = RGB_TOP_FINAL + RGB_PIXEL_HEIGHT * (rgb_cropping_coordinates_px["bottom"] - rgb_cropping_coordinates_px["top"]) # m
    RGB_RIGHT_FINAL = RGB_LEFT_FINAL + RGB_PIXEL_WIDTH *  (rgb_cropping_coordinates_px["right"] - rgb_cropping_coordinates_px["left"])# m

    print('RGB final coordinates in meters: final top = ' + str(RGB_TOP_FINAL) + ', final bottom = ' + str(RGB_BOTTOM_FINAL) + ', final left = ' + str(RGB_LEFT_FINAL) + ', final right = ' + str(RGB_RIGHT_FINAL))

    if save_image is not None:
        # crop the RGB orthomosaic to cover the same area as the thermal 
        # orthomosaic, converting data to uint8
        rgb_orthomosaic = rgb_bands[rgb_cropping_coordinates_px["top"] : rgb_cropping_coordinates_px["bottom"], rgb_cropping_coordinates_px["left"] : rgb_cropping_coordinates_px["right"]].astype('uint8') 
        print('orthomosaic shape after cropping to match thermal =', rgb_orthomosaic.shape)
        # save RGB orthomosaic as numpy array
        np.save(f'{save_image}', rgb_orthomosaic)

    return rgb_cropping_coordinates_px


def process_lidar_orthomosaic_aligned_with_thermal(lidar_tiff_path: str, thermal_cropping_coordinates_m: dict, thermal_orthomosaic_shape, thermal_interval: int, save_image: bool):
    """
    Processes LiDAR image, normalizing pixels (sets minimum to 0 and
    scales all other pixel values accordingly), and cropping to just the
    region of the thermal tiff containing data (non-zero pixels)--this aligns
    the RGB orthomosaic with the thermal orthomosaic previously processed.  
    Saves the processed RGB image as a numpy array.

    NOTE: thermal image must be the first to be processed in order to align
    the rest of the images (the coordinates from where it is cropped from the 
    original image are used to crop the other images. This assumes all the images
    (thermal, RGB, and LiDAR have the same dimension and were originally aligned)

    Parameters:
    -----------
    lidar_tiff_path: str
        absolute path to where the lidar tiff to be processed is stored
    thermal_cropping_coordinates_m: dict
        keys: THERMAL_TOP_FINAL, THERMAL_LEFT_FINAL, THERMAL_BOTTOM_FINAL, 
        THERMAL_RIGHT_FINAL

        values: coordinates in *meters* corresponding to the original thermal
            image that the processed image is cropped from.
            key and its corresponding value:
            TOP: the coordinate from the original thermal image where the 
                top edge of the cropped thermal image is
            LEFT: the coordinate from the original thermal image where the 
                left-most edge of the cropped thermal image is
            BOTTOM: the coordinate from the original thermal image where the 
                bottom edge of the cropped thermal image is
            RIGHT: the coordinate from the original thermal image where the 
                right-most edge of the cropped thermal image is
        NOTE: this is outputtd by process_thermal_orthomosaic

    save_image: str or None
        if not None: the processed mask image is saved as a numpy array to
        the file named specified by save_image
        otherwise: processed mask image not saved
    """
    # process LiDAR orthomosaic
    print('LiDAR orthomosaic')
    LIDAR_INTERVAL = 80 # width of cropped LiDAR images in pixels
    LIDAR_STRIDE = 20 # overlap of cropped LiDAR images in pixels

    lidar_dataset = gdal.Open(lidar_tiff_path)

    LIDAR_NUM_ROWS = lidar_dataset.RasterYSize # pixels
    LIDAR_NUM_COLS = lidar_dataset.RasterXSize # pixels

    LIDAR_PIXEL_HEIGHT = lidar_dataset.GetGeoTransform()[5] # m
    LIDAR_PIXEL_WIDTH = lidar_dataset.GetGeoTransform()[1] # m

    LIDAR_TOP = lidar_dataset.GetGeoTransform()[3] # m
    LIDAR_LEFT = lidar_dataset.GetGeoTransform()[0] # m
    LIDAR_BOTTOM = LIDAR_TOP + LIDAR_PIXEL_HEIGHT * LIDAR_NUM_ROWS # m
    LIDAR_RIGHT = LIDAR_LEFT + LIDAR_PIXEL_WIDTH * LIDAR_NUM_COLS # m
    print('top = ' + str(LIDAR_TOP) + ', bottom = ' + str(LIDAR_BOTTOM) + ', left = ' + str(LIDAR_LEFT) + ', right = ' + str(LIDAR_RIGHT))

    lidar_band = (lidar_dataset.GetRasterBand(1)).ReadAsArray(0, 0, LIDAR_NUM_COLS, LIDAR_NUM_ROWS)
    lidar_orthomosaic_masked = np.ma.masked_equal(lidar_band, -9999).filled(0)
    print('original orthomosaic shape =', lidar_orthomosaic_masked.shape)

    # get thermal data for aligning lidar image to thermal orthomosaic through cropping
    THERMAL_TOP_FINAL = thermal_cropping_coordinates_m["THERMAL_TOP_FINAL"]
    THERMAL_BOTTOM_FINAL = thermal_cropping_coordinates_m["THERMAL_BOTTOM_FINAL"]
    THERMAL_LEFT_FINAL = thermal_cropping_coordinates_m["THERMAL_LEFT_FINAL"]
    THERMAL_RIGHT_FINAL = thermal_cropping_coordinates_m["THERMAL_RIGHT_FINAL"]
    THERMAL_INTERVAL = thermal_interval

    # get coordinates (in pixels) to crop the lidar image at (crop at the same
    # meters coords as for the thermal image and then convert to pixel coords
    # in the thermal images)
    lidar_top = int((THERMAL_TOP_FINAL - LIDAR_TOP) / LIDAR_PIXEL_HEIGHT)
    lidar_bottom = int(lidar_orthomosaic_masked.shape[0] + (THERMAL_BOTTOM_FINAL - LIDAR_BOTTOM) / LIDAR_PIXEL_HEIGHT)
    lidar_left = int((THERMAL_LEFT_FINAL - LIDAR_LEFT) / LIDAR_PIXEL_WIDTH)
    lidar_right = int(lidar_orthomosaic_masked.shape[1] + (THERMAL_RIGHT_FINAL - LIDAR_RIGHT) / LIDAR_PIXEL_WIDTH)

    # calculate the new coordinates (in meters) of the edges of the cropped thermal image
    LIDAR_TOP_FINAL = LIDAR_TOP + lidar_top * LIDAR_PIXEL_HEIGHT # m
    LIDAR_LEFT_FINAL = LIDAR_LEFT + lidar_left * LIDAR_PIXEL_WIDTH # m
    LIDAR_BOTTOM_FINAL = LIDAR_TOP_FINAL + LIDAR_PIXEL_HEIGHT * (lidar_bottom - lidar_top) # m
    LIDAR_RIGHT_FINAL = LIDAR_LEFT_FINAL + LIDAR_PIXEL_WIDTH *  (lidar_right - lidar_left)# m

    print('LIDAR final coordinates in meters: final top = ' + str(LIDAR_TOP_FINAL) + ', final bottom = ' + str(LIDAR_BOTTOM_FINAL) + ', final left = ' + str(LIDAR_LEFT_FINAL) + ', final right = ' + str(LIDAR_RIGHT_FINAL))

    if save_image is not None:
        # crop lidar image so that it is aligned with thermal image
        lidar_orthomosaic = lidar_orthomosaic_masked[lidar_top : lidar_bottom, lidar_left : lidar_right].astype('uint8') # crop the LiDAR orthomosaic to cover the same area as the thermal orthomosaic
        print(f"initial lidar cropped coordinates: {lidar_orthomosaic.shape}")
        print('orthomosaic shape after cropping to match thermal =', lidar_orthomosaic.shape)
        np.save(f'{save_image}', lidar_orthomosaic) # save LiDAR orthomosaic as numpy array


def process_mask_aligned_with_rgb(mask_tiff_path: str, rgb_cropping_coordinates_px: dict, save_image: str):
    """
    Processes mask tiff corresponding to an RGB image, cropping to just 
    the region of the RGB tiff in the processed RGB mosaic (which should
    have been processed to align with the thermal orthomosaic previously
    processed, see 'process_rgb_orthomosaic_aligned_with_thermal')
    Saves the processed mask tiff image as a int16 numpy array if save_image.

    NOTE: assumes that the mask tiff being processed was rasterized from a 
    shapefile aligned with the RGB tiff image using the extent and resolution 
    of the RGB image in order to align it with the RGB image.  
    Additional assumptions about the RGB image's cropping are in 
    'process_rgb_orthomosaic_aligned_with_thermal'.

    Parameters:
    -----------
    mask_tiff_path: str
        absolute path to where the mask tiff to be processed is stored
    
    rgb_cropping_coordinates_px: dict
        key: value pairs
        "top": coordinate (in *pixels*) where the top edge of the cropped RGB
            image is relative to the original RGB image
        "bottom": coordinate (in *pixels*) where the bottom edge of the cropped RGB
            image is relative to the original RGB image
        "left": coordinate (in *pixels*) where the left edge of the cropped RGB
            image is relative to the original RGB image
        "right": coordinate (in *pixels*) where the right edge of the cropped RGB
            image is relative to the original RGB image
        NOTE: these are outputted from 'process_rgb_orthomosaic_aligned_with_thermal' 
        and can be used to crop a mask tiff rasterized (from a shapefile 
        aligned with the RGB tiff image) using the extent and resolution 
        of the RGB image to align it with the RGB image

    save_image: str or None
        if not None: the processed mask image is saved as a numpy array to
        the file named specified by save_image
        otherwise: processed mask image not saved
    """
    print("mask orthomosaic")
    # converts the tiff to a Dataset object
    mask_dataset = gdal.Open(mask_tiff_path) 

    mask_num_rows = mask_dataset.RasterYSize # pixels
    mask_num_cols = mask_dataset.RasterXSize # pixels
    # read in the data from the mask tiff as a np.array with data type int16
    # NOTE: this data type is made on the assumption that the mask was 
    # rasterized using int16 for pixel data type; reading 1 band is because
    # of assumption that the tiff data is in grayscale
    mask_band = ((mask_dataset.GetRasterBand(1)).ReadAsArray(0, 0, mask_num_cols, mask_num_rows).astype(np.int16))
    print(f"read mask orthomosaic size: {mask_band.shape}")

    # Get nodata value from the GDAL band object
    no_data_value = mask_dataset.GetRasterBand(1).GetNoDataValue()
    print(f"No data value: {no_data_value}")
    # Fill no data values with 0 (they're the background)
    mask_orthomosaic_masked = np.ma.masked_equal(mask_band, no_data_value).filled(0)

    # crop the mask orthomosaic to cover the same region as the RGB orthomosaic
    mask_orthomosaic = mask_orthomosaic_masked[rgb_cropping_coordinates_px["top"] : rgb_cropping_coordinates_px["bottom"], rgb_cropping_coordinates_px["left"] : rgb_cropping_coordinates_px["right"]]

    if save_image is not None:
        print('orthomosaic shape after cropping to match RGB =', mask_orthomosaic.shape)
        # save mask orthomosaic as numpy array
        np.save(f'{save_image}', mask_orthomosaic) 

def align_process_tiffs(thermal_tiff_path: str, rgb_tiff_path: str, lidar_tiff_path: str, mask_tiff_path: str, save_thermal: str, save_rgb: str, save_lidar: str, save_mask: str):
    """
    Processes all of the tiffs corresponding to each type of data (thermal, 
    RGB, LiDAR, and masks) into numpy arrays, making sure that they are all
    aligned.

    NOTE: everything is aligned with respect to the processed thermal tiff.

    Parameters:
    -----------
    thermal_tiff_path: str
        absolute path to where the thermal tiff to process is stored
    rgb_tiff_path: str or None
        absolute path to where the rgb tiff to process is stored
        if None: not processing an rgb tiff, NOTE: then cannot process mask
    lidar_tiff_path: str or None
        absolute path to where the lidar tiff to process is stored
        if None: not processing a lidar tiff
    mask_tiff_path: str or None
        absolute path to where the mask tiff to process is stored, NOTE:
        assumed to be from shapefile made overlaid on rgb tiff and rasterized
        using same extent and resolution as rgb tiff
        if None: not processing a mask tiff

    save_<image_modality>: str or None
        if None: not saving the processed tiff for the <image_modality> data
        to a saved numpy array

        otherwise: absolute path to directory to store the processed
        <image_modality> tiff to.  Processed tiff is in the form of a numpy
        array, file name will be <image_modality>-orthomosaic-matrix.npy
    """
    thermal_cropping_coordinates_m, thermal_orthomosaic_shape, thermal_interval = process_thermal_orthomosaic(thermal_tiff_path, save_thermal)
    if rgb_tiff_path is not None:
        rgb_cropping_coordinates_px = process_RGB_orthomosaic_aligned_with_thermal(rgb_tiff_path, thermal_cropping_coordinates_m, save_rgb)
    if ((rgb_tiff_path is not None) and (mask_tiff_path is not None)):
        process_mask_aligned_with_rgb(mask_tiff_path, rgb_cropping_coordinates_px, save_mask)
    if lidar_tiff_path is not None:
        process_lidar_orthomosaic_aligned_with_thermal(lidar_tiff_path, thermal_cropping_coordinates_m, thermal_orthomosaic_shape, thermal_interval, save_lidar)
    
    

if __name__ == '__main__':
    # process system arguments
    print("System arguments")
    for x in sys.argv:
        print(x)

    thermal_tiff_path = sys.argv[1]
    rgb_tiff_path = sys.argv[2]
    if rgb_tiff_path == "None":
        rgb_tiff_path = None
    lidar_tiff_path = sys.argv[3]
    if lidar_tiff_path == "None":
        lidar_tiff_path = None
    mask_tiff_path = sys.argv[4]
    if mask_tiff_path == "None":
        mask_tiff_path = None
    save_thermal = sys.argv[5]
    if save_thermal == "None":
        save_thermal = None
    save_rgb = sys.argv[6]
    if save_rgb == "None":
        save_rgb = None
    save_lidar = sys.argv[7]
    if save_lidar == "None":
        save_lidar = None
    save_mask = sys.argv[8]
    if save_mask == "None":
        save_mask = None
    
    # process each of the tiffs passed in into numpy arrays, aligning them
    # with each other, and saving them to the specified file path (if saving)
    align_process_tiffs(thermal_tiff_path, rgb_tiff_path, lidar_tiff_path, mask_tiff_path, save_thermal, save_rgb, save_lidar, save_mask)
