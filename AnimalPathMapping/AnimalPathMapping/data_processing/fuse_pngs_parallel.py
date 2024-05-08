'''
fuse by Samantha Marks and Lucia Gordon
inputs: thermal, RGB, and LiDAR image arrays
outputs: fused image arrays for thermal-RGB, (TODO: removed: thermal-LiDAR, and RGB-LiDAR)

TODO: make parallelized version of this which grabs by image id (path to image with same id for each of the modalities as input)
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
def fuse_parallel(image_tiles_path: str, output_path: str, ident: str, modality1: str, modality2: str, modality3: str=None):
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

    ident: str
        the {i} from image_tiles_path file directory structure (id of the
        image to fuse across modalities, assumes the same image has the same
        id across its representation in the different modalities)

    Returns:
    fused_arrays: list
        list of numpy arrays, with each numpy array corresponding to a fused
        image tile (image tile that has been fused across multiple modalities,
        where the image tile covers the same region in each modality)
        here contains just one element
    """
    fused_arrays = []

    image1 = Image.open(f'{image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{ident}.png')
    image2 = Image.open(f'{image_tiles_path}/{modality2}-tiles/png-images/{modality2}-{ident}.png')
    fused = Image.blend(image1, image2, 0.5)

    if modality3 is not None:
        image3 = Image.open(f'{image_tiles_path}/{modality3}-tiles/png-images/{modality3}-{ident}.png')
        fused = Image.blend(fused, image3, 1/3)

    # save image to png
    fused.save(f'{output_path}/image-{ident}.png', 'PNG')
    
    fused_arrays.append(imread(f'{output_path}/image-{ident}.png'))

    return fused_arrays

def fuse_weighted_parallel(image_tiles_path: str, output_path: str, ident: str, modality1: str, modality2: str, modality2_weight: float, modality3: str=None, modality3_weight: float=None):
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

    ident: str
        the {i} from image_tiles_path file directory structure (id of the
        image to fuse across modalities, assumes the same image has the same
        id across its representation in the different modalities)

    TODO modality

    modality2_weight: int
        weighting of modality 2 image (between 0 and 1), 0.2 means resulting
        combination of images 1 and 2 puts 0.8 weight on image 1, 0.2 weight
        on image 2 to get the resulting combination of the 2

    Returns:
    fused_arrays: list
        list of numpy arrays, with each numpy array corresponding to a fused
        image tile (image tile that has been fused across multiple modalities,
        where the image tile covers the same region in each modality)
        here contains just one element
    """
    fused_arrays = []

    image1 = Image.open(f'{image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{ident}.png')
    image2 = Image.open(f'{image_tiles_path}/{modality2}-tiles/png-images/{modality2}-{ident}.png')
    fused = Image.blend(image1, image2, modality2_weight)

    if modality3 is not None:
        image3 = Image.open(f'{image_tiles_path}/{modality3}-tiles/png-images/{modality3}-{ident}.png')
        fused = Image.blend(fused, image3, modality3_weight)

    # save image to png
    fused.save(f'{output_path}/image-{ident}.png', 'PNG')
    
    fused_arrays.append(imread(f'{output_path}/image-{ident}.png'))

    return fused_arrays

if __name__ == '__main__':
    # process sytem arguments, assumes path to an rgb image is first arg
    image_path = sys.argv[1]
    rgb_weight = float(sys.argv[2])
    print(f"Inputted image path is: {image_path}")
    print(f"Inputted RGB weight is: {rgb_weight}")
    # assumes path looks like this: {image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{i}.png'
    # and that image_tiles_path contains 'rgb-tiles', 'thermal-tiles' and 'lidar-tiles' folders
    # TODO temporarily assumes not using lidar images
    # and same image across the 3 modalities ends with the same id, 'i'.
    # TODO later add sanity check that does look like this (check i is numeric, 
    # check image_tiles_path ends with the rest of the file structure, etc)

    # get image_tiles_path and the identifier i
    file_base, _ = os.path.splitext(image_path) # remove file extension
    file_base, file_name = os.path.split(file_base) # split on file name
    ident = file_name.split("-")[1] # file name ends with id after '-', as in '{modality1}-{i}', i is id
    print(f"Image identifer is: {ident}")
    file_base, _ = os.path.split(file_base) # remove png-images/
    image_tiles_path, _ = os.path.split(file_base) # remove {modality1}-tiles/ so left with {image_tiles_path}
    print(f"Root image tiles path that the inputted image path was stored in is: {image_tiles_path}")
    
    # this code assumes using all 3 image modalities and want all methods of fusing
    if not os.path.exists(f'{image_tiles_path}/fused'):
        try:
            os.mkdir(f'{image_tiles_path}/fused')
        except:
            # assuming other function call tried to make the dir
            pass

    # TODO remove (this is thermal-rgb 50-50) once have a set ratio for thermal and RGB
    # output_path = f'{image_tiles_path}/fused/tr-fused'
    # if not os.path.exists(output_path):
    #     try:
    #         os.mkdir(output_path)
    #     except:
    #         # assuming other function call tried to make the dir
    #         pass
    # np.save(f'{output_path}/tr-fused-all.npy', fuse_parallel(image_tiles_path, output_path, ident, 'thermal', 'rgb'))

    # Fuse together thermal and RGB image tile, putting 80% weight on RGB
    thermal_weight = int(round(((1 - rgb_weight) * 100)))
    output_path = f'{image_tiles_path}/fused/tr-fused-{thermal_weight}-{int(rgb_weight * 100)}'
    if not os.path.exists(output_path):
        try:
            os.mkdir(output_path)
        except:
            # assuming other function call tried to make the dir
            pass
    np.save(f'{output_path}/tr-fused-all.npy', fuse_weighted_parallel(image_tiles_path, output_path, ident, 'thermal', 'rgb', rgb_weight))
    print(f'thermal-RGB {thermal_weight}-{100-thermal_weight} fusing done')
    

    # TODO take this out or make lidar optional command argument
    # output_path = f'{image_tiles_path}/fused/tl-fused'
    # if not os.path.exists(output_path):
    #     try:
    #         os.mkdir(output_path)
    #     except:
    #         # assuming other function call tried to make the dir
    #         pass
    # np.save(f'{output_path}/tl-fused-all.npy', fuse_parallel(image_tiles_path, output_path, ident, 'thermal', 'lidar'))
    # print('thermal-LiDAR fusing done')
    # output_path = f'{image_tiles_path}/fused/rl-fused'
    # if not os.path.exists(output_path):
    #     try:
    #         os.mkdir(output_path)
    #     except:
    #         # assuming other function call tried to make the dir
    #         pass
    # np.save(f'{output_path}/rl-fused-all.npy', fuse_parallel(image_tiles_path, output_path, ident, 'rgb', 'lidar'))
    # print('RGB-LiDAR fusing done')
    # output_path = f'{image_tiles_path}/fused/trl-fused'
    # if not os.path.exists(output_path):
    #     try:
    #         os.mkdir(output_path)
    #     except:
    #         pass
    # np.save(f'{output_path}/trl-fused-all.npy', fuse_parallel(image_tiles_path, output_path, ident, 'thermal', 'rgb', 'lidar'))
    # print('thermal-RGB-LiDAR fusing done')
    # output_path = f'{image_tiles_path}/fused/trl-fused-(20-80)-15'
    # if not os.path.exists(output_path):
    #     try:
    #         os.mkdir(output_path)
    #     except:
    #         # assuming other function call tried to make the dir
    #         pass
    # np.save(f'{output_path}/trl-fused-all.npy', fuse_weighted_parallel(image_tiles_path, output_path, ident, 'thermal', 'rgb', 0.8, 'lidar', 0.15))
    # print('thermal-RGB (20-80)-15 fusing done')
    

