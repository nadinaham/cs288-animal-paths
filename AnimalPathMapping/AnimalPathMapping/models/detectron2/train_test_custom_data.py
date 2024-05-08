# import torch, detectron2
# !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

import torch, detectron2
from detectron2.utils.logger import setup_logger
import logging

setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

# For mask conversion to dictionary
import pycocotools.mask as mask_util

# For train-test split
import random

# For LossEvalHook
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.data import build_detection_test_loader
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime

# Plotting
import json
import matplotlib.pyplot as plt

"""Create test-train split based on image IDs"""
def train_val_test_split(all_ids, train_frac=0.6, val_frac=0.2):
    random.shuffle(all_ids)

    train_end = int(train_frac * len(all_ids))
    val_end = train_end + int(val_frac * len(all_ids))
    
    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]
    
    return train_ids, val_ids, test_ids

"""Create custom dataset dictionaries given image and mask directories"""
def get_custom_dicts(ids, img_dir, mask_dir):
    dataset_dicts = []

    # List all PNG files in img_dir to process as dataset images. Filter for the ids asked for
    image_files = [f for f in os.listdir(img_dir) if f.startswith('image') and f.endswith('.png') and f.split('-')[-1].split('.')[0] in ids]

    # Creating a set of IDs from image files to ensure matching mask files exist
    image_ids = set(f.split('-')[-1].split('.')[0] for f in image_files)

    for idx, image_filename in enumerate(image_files):
        # Get masks to current image
        image_id = image_filename.split('-')[-1].split('.')[0]

        mask_folder_path = os.path.join(mask_dir, f"image_{image_id}_segmentation_masks")

        # Check if mask directory exists
        if not os.path.exists(mask_folder_path):
            # Skip this image and proceed to the next
            print(f"Skipping image {image_filename} as mask directory does not exist.")
            continue  # Skip the current iteration

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

        mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.npy')]

        # Initialize a list to store object annotations
        objs = []

        # MASK MANIPULATION
        for mask_filename in mask_files:
            mask_path = os.path.join(mask_folder_path, mask_filename)
            mask = np.load(mask_path)

            # Replace with COCO dictionary tools
            mask = np.array(mask, dtype=np.uint8)
            mask = np.asfortranarray(mask)
            encoded_mask = mask_util.encode(mask)
            
            bbox = mask_util.toBbox(encoded_mask).flatten().tolist()
            area = mask_util.area(encoded_mask).item()
            
            obj = {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": encoded_mask,  # RLE
                "category_id": 0,  # assuming a single class
                "area": area  # the area of the mask
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Handle CLAs
import sys
import subprocess

if len(sys.argv) != 4:
    print("Usage: python script.py img_dir mask_dir output_dir")
    sys.exit(1)

img_dir = sys.argv[1]
mask_dir = sys.argv[2]
output_dir = sys.argv[3]

# Error handling
if not os.path.isdir(img_dir):
    print(f"Error: The specified image directory does not exist")
    exit(0)

if not os.path.isdir(mask_dir):
    print(f"Error: The specified mask directory does not exist")
    exit(0)

print("CHECKPOINT: Directories loaded")

# Train-test split by image ids
all_image_ids = [os.path.splitext(f)[0].split('-')[-1] for f in os.listdir(img_dir)]

train_ids, val_ids, test_ids = train_val_test_split(all_image_ids)

# Registering the datasets
for name, dataset in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    DatasetCatalog.register("custom_" + name, lambda d=name, img=img_dir, msk=mask_dir, ids=dataset: get_custom_dicts(ids, img, msk))
    MetadataCatalog.get("custom_" + name).set(thing_classes=["object"])

print("CHECKPOINT: Train-validation-test split complete")

custom_metadata = MetadataCatalog.get("custom_train")

# Update the configuration to use the new dataset
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.VAL = ("custom_val",)
cfg.DATASETS.TEST = ("custom_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3500
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 
cfg.INPUT.MASK_FORMAT = "bitmask" # To tell the trainer to not expect polygons
cfg.OUTPUT_DIR = output_dir

print("CHECKPOINT: Configurations loaded")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

print("CHECKPOINT: Model trained")

print("CHECKPOINT: starting evaluation")

# -------------------------------

# Saving model + weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

dataset_dicts = get_custom_dicts(test_ids, img_dir, mask_dir)

# Method 1: Randomly sample dictionaries to visualize
num_sample = 50
sampled_dicts = random.sample(dataset_dicts, num_sample)
# sampled_dicts = dataset_dicts

for idx, d in enumerate(sampled_dicts):
    # Read the image
    im = cv2.imread(d["file_name"])
    # Predict using the trained model
    outputs = predictor(im)
    # Visualize the predictions
    v = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Convert BGR to RGB for visualization
    vis_image = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    # Save the visualized output to a file
    output_filename = f"sample_image_{idx}.png"  # Name the output file
    directory = output_dir
    output_path = os.path.join(directory, output_filename)
    cv2.imwrite(output_path, vis_image)
    print(f"wrote image {idx} to: {output_path}")
 
# Method 2: Evaluate performance using AP metric implemented in COCO API

evaluator = COCOEvaluator("custom_test", ("bbox",), False, output_dir=f"{output_dir}/inference")
val_loader = build_detection_test_loader(cfg, "custom_test")
results = inference_on_dataset(trainer.model, val_loader, evaluator)
print_csv_format(results)

# -------------------------------

f = open(cfg.OUTPUT_DIR + 'config.yml', 'w')
f.write(cfg.dump())
f.close()

print("CHECKPOINT: Config saved")

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(cfg.OUTPUT_DIR + '/metrics.json')

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.legend(['total_loss'], loc='upper left')
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_metrics.png'))

print("CHECKPOINT: Data plotted")