'''
rotate_parallel by Samantha Marks
input: path to image .png file to be rotated, assumes looks like {image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{i}.png'
        boolean where 'True' means that the masks corresponding with the inputted image with be rotated too
outputs: for an image with id 'i' that gets rotated 'd' degrees, it will get a new
    id label called 'l', where l = i + d % 4, where 4 is the number of rotations, and % is modulo
    the rotated image, 'l', will be stored in: '{image_tiles_path}/rotated/{modality1}-tiles/png-images/{modality1}-{i}-rot-{d}-{l}.png'
    
    If rotating the masks too, the masks corresponding to image 'l' (which are all the masks corresponding to 
    image 'i' rotated 'd' degrees) will be as stored as .npy files (for training a model) in: 
        '{image_tiles_path}/rotated/path-masks-finished/image_{l}_i-rot-{d}_segmentation_masks'
    and as .png files (for visualization) in:
        '{image_tiles_path}/rotated/path-masks-finished/image_{l}_i-rot-{d}_visualization_masksv'
    where each mask m will be named:
        'image_l_i-rot-{d}_{seg/vis}_mask.{npy/png}' (option a of a/b if segmentation mask, option b for visualization mask)


TODO this currently assumes one image modality, make option for including thermal in addition to RGB
'''

# imports
import os
import numpy as np
from PIL import Image
from shutil import rmtree
from cv2 import imread
from sys import argv
import sys
import glob


def rotate_image(image_path: str, degree: str, output_path: str):
    """
    Rotates the image by the specified degree

    Parameters:
    -----------
    image_path: str
        absolute path to the image file (as a .png file) to be rotated
    """
    # Use PIL to rotate the image the specified number of degrees and save it
    im = Image.open(image_path)
    im_rotated = im.rotate(degree)
    im_rotated.save(output_path)

def rotate_mask(mask_path: str, degree: str, output_npy_path: str, output_png_path: str):
    """
    Rotates the mask by the specified degree

    Parameters:
    -----------
    mask_path: str
        absolute path to the mask file (as a .npy file) to be rotated
        TODO: put in requirements about type of this .npy
    """
    # Load mask numpy array
    mask = np.load(mask_path, allow_pickle=True)

    # Read image into PIL
    mask_im = Image.fromarray(mask)
    # Then use PIL to rotate by specified number of degrees
    mask_rotated = mask_im.rotate(degree)
    # Then cnvt back to .npy array and store in output_npy_path
    rot_mask_arr = np.array(mask_rotated).astype(np.uint8)
    np.save(output_npy_path, rot_mask_arr)
    # And also save as png and store in output_png_path, need to covert
    # into 3-channel RGB beforehand
    # TODO make conversion to RGB a helper function
    lookup = np.array([[0,0,0], [255,255,255]])
    rgb = lookup[rot_mask_arr].astype(np.uint8)
    img = Image.fromarray(rgb, 'RGB')
    img.save(output_png_path)




if __name__ == '__main__':
    # process sytem arguments, assumes path to an rgb image is first arg
    image_path = sys.argv[1]
    do_mask_rotate = sys.argv[2] # is True or False
    mask_input_folders = sys.argv[3].split(",")
    print(f"Inputted image path is: {image_path}")
    print(f"Mask input folders are: {mask_input_folders}")
    # assumes path looks like this: {image_tiles_path}/{modality1}-tiles/png-images/{modality1}-{i}.png'
    # and that image_tiles_path contains 'rgb-tiles' [TODO Correct later: 
    # [, 'thermal-tiles' and 'lidar-tiles']], and 'path-masks-finished' folders
    # TODO temporarily assumes not using thermal images, correct this later
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
    

    rotated_image_output_folder = f'{image_tiles_path}/rotated/rgb-tiles/png-images'
    if not os.path.exists(rotated_image_output_folder):
        try:
            os.makedirs(rotated_image_output_folder)
        except:
            # assuming other function call tried to make the dir
            pass
    
    # rotate image 4 times, by 90 degrees each time
    rot_deg = 90 # degree to rotate by
    for deg in range(0, 360, rot_deg):
        # {modality1}-{i}-rot-{d}-{l}.png'
        # rotated_image_output_path = f"{rotated_image_output_folder}/rot-{deg}-rgb-{ident}.png"
        # new ident is calculated this way to make sure no overlapping new ids when 
        # processing all images in parallel
        new_ident = int(((360/rot_deg) * int(ident)) + (deg / rot_deg))
        print(f"New image identity for image {ident} rotated {deg} degrees: {new_ident}")
        rotated_image_output_path = f"{rotated_image_output_folder}/rgb-{ident}-rot-{deg}-{new_ident}.png"
        rotate_image(image_path, deg, rotated_image_output_path)

    # rotate mask corresponding to image if specified to
    if do_mask_rotate == "True":
        for mask_folder in mask_input_folders:
            if mask_folder != "None":
                mask_output_path_ending = f'path-masks-finished/{mask_folder}'
                mask_seg_path_input_ending = f'path-masks-finished/{mask_folder}/image_{ident}_segmentation_masks'
                # mask_vis_path_ending = f'path-masks-finished/{mask_folder}/image_{ident}_visualization_masksv'

            else:
                mask_output_path_ending = f'path-masks-finished'
                # mask_seg_path_ending = f'path-masks-finished/image{ident}_segmetnation_masks'
                # mask_vis_path_ending = f'path-masks-finished/image_{ident}_visualization_masksv'

            # rotated_mask_output_folder = f'{image_tiles_path}/rotated/{mask_seg_path_ending}'
            # rotated_mask_vis_output_folder = f'{image_tiles_path}/rotated/{mask_vis_path_ending}'
            rotated_mask_output_folder = f'{image_tiles_path}/rotated/{mask_output_path_ending}'

            print(f"Outer folder for outputted rotated mask .npy files: {rotated_mask_output_folder}")
            # print(f"Output folder for rotated mask .png files: {rotated_mask_vis_output_folder}")


            if not os.path.exists(rotated_mask_output_folder):
                try:
                    os.makedirs(rotated_mask_output_folder)
                except:
                    # assuming other function call tried to make the dir
                    pass
            # if not os.path.exists(rotated_mask_vis_output_folder):
            #     try:
            #         os.makedirs(rotated_mask_vis_output_folder)
            #     except:
            #         # assuming other function call tried to make the dir
            #         pass
            # get all the masks in the directory corresponding to the image, assumes that mask
            # directory path is: '{image_tiles_path}/path-masks-finished/image_{ident}-segmentation_masks/'
            mask_input_dir = f'{image_tiles_path}/{mask_seg_path_input_ending}'
            # mask_dir = os.path.join(image_tiles_path, "path-masks-finished", f"image_{ident}_segmentation_masks")
            mask_files = glob.glob(f"{mask_input_dir}/*.npy")
            print(f"Mask files being processed: {mask_files}")

            # rotate each mask 4 times, by 90 degrees each time
            # rotate image 4 times, by 90 degrees each time
            for mask_file in mask_files:
                # get mask number
                file_base, _ = os.path.splitext(mask_file) # remove file extension
                file_base, file_name = os.path.split(file_base) # split on file name
                # mask_number = file_name.split("_")[-1] # file name ends with id after '_', assumes file name is: 'image_{ident}_seg_mask_{number}.npy'
                # TODO take out, esp when switch to merged masks
                if mask_folder == "masks_sep":
                    mask_number = file_name.split("k")[-1] # temp fix for ending with seg_mask{number}.npy, TODO take out
                else:
                    mask_number = file_name.split("_")[-1]
                print(f"mask number {mask_number} for image id {ident} is being processed, this mask belongs to file: {mask_file}")
                rot_deg = 90 # degree to rotate by
                for deg in range(0, 360, rot_deg):
                    # {modality1}-{i}-rot-{d}-{l}.png'
                    # rotated_image_output_path = f"{rotated_image_output_folder}/rot-{deg}-rgb-{ident}.png"
                    # new ident is calculated this way to make sure no overlapping new ids when 
                    # processing all images in parallel
                    new_ident = int(((360/rot_deg) * int(ident)) + (deg / rot_deg))
                    # '{image_tiles_path}/rotated/path-masks-finished/image_{l}_{modality1}-i-rot-{d}_visualization_masksv'
                    # where each mask m will be named:
                    #     'image_l_i-rot-{d}_{seg/vis}_mask.{npy/png}' (option a of a/b if segmentation mask, option b for visualization mask)
                    # rotated_mask_output_path = f"{rotated_mask_output_folder}/image_{ident}_rot_{deg}_seg_mask_{mask_number}.npy"
                    # rotated_mask_visualization_path = f"{rotated_mask_vis_output_folder}/image_{ident}_rot_{deg}vis_mask_{mask_number}.png"
                    #  '{image_tiles_path}/rotated/path-masks-finished/image_{l}_i-rot-{d}_segmentation_masks'
                    rotated_mask_seg_folder = f"{rotated_mask_output_folder}/image_{new_ident}_{ident}-rot-{deg}_segmentation_masks"
                    rotated_mask_visualization_folder = f"{rotated_mask_output_folder}/image_{new_ident}_{ident}-rot-{deg}_visualization_masksv"
                    
                    rotated_mask_output_path = f"{rotated_mask_seg_folder}/image_{new_ident}_{ident}-rot-{deg}_seg_mask_{mask_number}.npy"
                    # rotated_mask_visualization_path = f"{rotated_mask_vis_output_folder}/image_{new_ident}_{ident}_rot-{deg}_vis_mask_{mask_number}.png"
                    rotated_mask_visualization_path = f"{rotated_mask_visualization_folder}/image_{new_ident}_{ident}-rot-{deg}_vis_mask_{mask_number}.png"

                    # make folders that the rotated mask .npy and .png will be saved to
                    if not os.path.exists(rotated_mask_seg_folder):
                        os.makedirs(rotated_mask_seg_folder)
                    if not os.path.exists(rotated_mask_visualization_folder):
                        os.makedirs(rotated_mask_visualization_folder)

                    rotate_mask(mask_file, deg, rotated_mask_output_path, rotated_mask_visualization_path)

