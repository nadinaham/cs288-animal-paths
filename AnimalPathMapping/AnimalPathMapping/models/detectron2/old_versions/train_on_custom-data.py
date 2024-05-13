# import torch, detectron2
# !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)
# Sam copied this from Yiyi and added some TODO's

import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

# Converting masks into dictionaries
import pycocotools.mask as mask_util

"""Create custom dataset dictionaries given image and mask directories"""
def get_custom_dicts(img_dir, mask_dir):

    dataset_dicts = []
    # List all PNG files in img_dir to process as dataset images.
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    for idx, image_filename in enumerate(image_files):
        # Initialise a dictionary to hold data about the image
        record = {}

        # Construct the full path to the image
        image_path = os.path.join(img_dir, image_filename)

        # Read image using OpenCV, and extract its dimensions
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Store relevant details
        record["file_name"] = image_path
        record["image_id"] = idx 
        record["height"] = height
        record["width"] = width

        # Get masks to current image
        mask_folder = os.path.splitext(image_filename)[0] + '_masks'
        mask_folder_path = os.path.join(mask_dir, mask_folder)
        mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.png')]
        
        # Initialize a list to store object annotations
        objs = []

        # Based on sample code version (may need modification)
        for mask_file in mask_files:
            assert not mask_file["region_attributes"]
            mask_file = mask_file["shape_attributes"]
            px = mask_file["all_points_x"]
            py = mask_file["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        
        # # Modified version (need to be tested and reviewed TODO)
        # for mask_filename in mask_files:
        #     # For each mask file, read the mask as a grayscale image, and use OpenCV to find contours which represent object boundaries
        #     mask_path = os.path.join(mask_folder_path, mask_filename)
        #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #     for contour in contours:
        #         # Approximate contours to polygons
        #         contour = cv2.approxPolyDP(contour, 3, True)

        #         # Flatten the array of points and convert it to a list
        #         segmentation = contour.flatten().tolist()
                
        #         # Skip any contours that are too small
        #         if len(segmentation) < 6:
        #             continue 
                
        #         # Calculate the bounding rectangle for the contour
        #         bbox = cv2.boundingRect(contour)

        #         # Store relevant info
        #         obj = {
        #             "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])],
        #             "bbox_mode": BoxMode.XYXY_ABS,
        #             "segmentation": [segmentation],
        #             "category_id": 0,  # assuming a single class
        #         }
        #         objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Registering the datasets (may need to modify the 'd')
for d in ["train", "val"]:
    img_dir = os.path.join("data/images", d)
    mask_dir = os.path.abspath("data/masks", d)
    DatasetCatalog.register("custom_" + d, lambda d=d, img=img_dir, msk=mask_dir: get_custom_dicts(img, msk))
    MetadataCatalog.get("custom_" + d).set(thing_classes=["object"])

# Not sure how to modify this line (TODO)
custom_metadata = MetadataCatalog.get("custom_train")

# Update the configuration to use the new dataset
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_train",) # Need to modify after train-test split TODO
cfg.DATASETS.TEST = ()  # Need to modify after train-test split TODO
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# TODO find way to save the model?  Is it to cfg.OUTPUT_DIR, what is OUTPUT_DIR?
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# save config
print(cfg.dump())
f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()

# Need to add code for inference and evaluation using the trained model TODO
# to get the predictor
# cfg.merge_from_file("config.yml")
# predictor = DefaultPredictor(cfg)
# can do inference from here