'''
fuse by Samantha Marks and Lucia Gordon
inputs: thermal, RGB, and LiDAR image arrays
outputs: fused image arrays for thermal-RGB, thermal-LiDAR, and RGB-LiDAR

NOTE: the parallelized version, in 'fuse_png_parallel.py' is the preferred
script to run, as it is SIGNIFICANTLY faster.
'''

# imports
import os
import numpy as np
from PIL import Image
from shutil import rmtree
from cv2 import imread
from sys import argv
import sys



# functions
def fuse(image_tiles_path: str, output_path: str, modality1: str, images1: list, modality2: str, images2: list, modality3: str=None, images3: list=None):
    """
    Combines the same image tile across two or three image modalities (of thermal,
    rgb, and lidar) into one png.

    NOTE: if all three modalities are used, LiDAR should be modality3

    Parameters:
    -----------
    image_tiles_path: str
        absolute path to folder containing image tiles to structure.  This
        was used as the 'output_folder' parameter in 'tile_orthomosaics'.
        File directory structure within it made by 'tile_orthomosaics' is:
            {output_folder}/{modality}-tiles/png-images'
                with .png images stored inside named {modality}-{i}.png
                for some integers i
            {output_folder}/{modality}-tiles/numpy-images'
                with a .npy file inside

    output_path: str
        absolute path to folder to store the fused image tiles in.  Should
        be a folder specific to the combination of modalities being fused.

    Returns:
    fused_arrays: list
        list of numpy arrays, with each numpy array corresponding to a fused
        image tile (image tile that has been fused across multiple modalities,
        where the image tile covers the same region in each modality)
    """
    fused_arrays = []

    for i in range(len(images1)):
        image1 = Image.open(f'{image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{i}.png')
        image2 = Image.open(f'{image_tiles_path}/{modality2}-tiles/png-images/{modality2}-{i}.png')
        fused = Image.blend(image1, image2, 0.5)
        # fused.save(f'{output_path}/', 'PNG')

        if modality3 is not None:
            image3 = Image.open(f'{image_tiles_path}/{modality3}-tiles/png-images/{modality3}-{i}.png')
            fused = Image.blend(fused, image3, 1/3)

        # save image to png
        fused.save(f'{output_path}/image-{i}.png', 'PNG')
        
        fused_arrays.append(imread(f'{output_path}/image-{i}.png'))

    return fused_arrays


def fuse_images(image_tiles_path: str, options: list):
    """
    Parameters:
    -----------
    image_tiles_path: str
        absolute path to directory containing all of the image tiles across
        all of the modalities to fuse.  This was the 'output_folder' argument
        to 'tile_orthomosaic.py', and this script stored image tiles in the
        following directory structure:

        {output_folder}/{modality}-tiles/png-images'
                with .png images stored inside named {modality}-{i}.png
                for some integers i
            {output_folder}/{modality}-tiles/numpy-images'
                with a 'all-numpy-images.npy' file inside
        NOTE: there must be a subfolder for each modality to be fused

    options: list
        list where viable values are:
        'tr', 'tl', 'rl', 'trl'
        where t stands for thermal, r stands for rgb, l stands for lidar
    """    
    # Create directory to store fused image tiles in
    if not os.path.exists(f'{image_tiles_path}/fused'):
        os.mkdir(f'{image_tiles_path}/fused')

    # Load only the image tiles belongging to the modalities being fused
    load_thermal = False
    load_rgb = False
    load_lidar = False
    for option in options:
        if option == "tr":
            load_thermal = True
            load_rgb = True
        elif option == "tl":
            load_thermal = True
            load_lidar = True
        elif option == "rl":
            load_rgb = True
            load_lidar = True
        elif option == "trl":
            load_thermal = True
            load_rgb = True
            load_lidar = True
        else:
            raise Exception(f"Invalid option.  Inputted: {option}.  Valid options are 'tr', 'tl', rl', or 'trl'.")
    
    # load in the image modalities in use
    if load_thermal:
        thermal_images = np.load(f'{image_tiles_path}/thermal-tiles/numpy-images/all-numpy-images.npy')
    if load_rgb:
        rgb_images = np.load(f'{image_tiles_path}/rgb-tiles/numpy-images/all-numpy-images.npy')
    if load_lidar:
        lidar_images = np.load(f'{image_tiles_path}/lidar-tiles/numpy-images/all-numpy-images.npy')

    # fuse together images of the different modalities specified
    for option in options:
        if option == "tr":
            output_path = f'{image_tiles_path}/fused/tr-fused'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            np.save(f'{output_path}/tr-fused-all.npy', fuse(image_tiles_path, output_path, 'thermal', thermal_images, 'rgb', rgb_images))
            print('thermal-RGB fusing done')
        elif option == "tl":
            output_path = f'{image_tiles_path}/fused/tl-fused'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            np.save(f'{output_path}/tl-fused-all.npy', fuse(image_tiles_path, output_path, 'thermal', thermal_images, 'lidar', lidar_images))
            print('thermal-LiDAR fusing done')
        elif option == "rl":
            output_path = f'{image_tiles_path}/fused/rl-fused'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            np.save(f'{output_path}/rl-fused-all.npy', fuse(image_tiles_path, output_path, 'rgb', rgb_images, 'lidar', lidar_images))
            print('RGB-LiDAR fusing done')
        else: # option is trl
            output_path = f'{image_tiles_path}/fused/trl-fused'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            np.save(f'{output_path}/trl-fused-all.npy', fuse(image_tiles_path, output_path, 'thermal', thermal_images, 'rgb', rgb_images, 'lidar', lidar_images))
            print('thermal-RGB-LiDAR fusing done')


if __name__ == '__main__':
    # process system arguments (excluding script name)
    print("System arguments:")
    image_tiles_path = sys.argv[1]
    print(f"image tiles path: {image_tiles_path}")
    options = []
    for i in range(2, len(sys.argv)):
        options.append(sys.argv[i])

    
    # fuse image tiles across modalities as specified by options
    fuse_images(image_tiles_path, options)
