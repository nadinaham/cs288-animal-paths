"""
File created by Samantha Marks for processing animal path masks from tiled (png) images.
This ensures that each path within a tiled image gets its own individual mask, with one
mask per distinct path in each tiled image.
"""

# TODO incomplete and on the wrong track

def open_tiled_masks(tiled_maskpath):
    """
    Opens the file containing the mask, sliced into tiles corresponding
    to the tiled RGB orthomosaic, as processed by 'tile_orthomosaic.py'

    Parameters:
    -----------
    mask_path: path to .npy file containing list of mask tiles, as
    processed by 'tile_orthomosaic.py'

    Returns:
    --------
    mask_tiles: np.array
        array of mask tiles (which are each 2D numpy arrays whose values
        are the ids of the paths in the mask--each path 'p' of id 'pi'
        has its 'pixels' colored with value 'pi', each index of the array
        corresponds to a pixel at its same position in the mask tiff image)
    """
    mask_tiles = np.load(mask_path)
    return mask_tiles

def get_unique_masks(mask_path):
    """
    Gets the unique masks corresponding to individual masks within a file 
    containing masks.


    Parameters:
    -----------
    mask_path: path to .png (or .npy?) file containing mask

    Returns:
    --------
    mask_annotations_dict: dictionary where key is mask number, value
        is np array with 1's corresponding to pixels for an individual
        path mask, 0 for background

    """
    # TODO look up custom dataset example for how to order
    mask_annotations_dict = {}
    # get the unique values from the image
    # throw out 0
    # iterate through each unique value
        # turn these to white on black background (black out all other pixels)
        # save at key corresponding to mask in dict (TODO see custom dataset for what key should be)

    return mask_annotations_dict