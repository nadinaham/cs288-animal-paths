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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
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

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime

# Custom loss hook for calculating validation
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

# Custom trainer to apply the custom loss hook for validation
class CustomTrainer(DefaultTrainer):
    """
    Custom Trainer deriving from the "DefaultTrainer"

    Overloads build_hooks to add a hook to calculate loss on the test set during training.
    """

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD, # Frequency of calculation - every 100 iterations here
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        return hooks

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
    image_files = [f for f in os.listdir(img_dir) if f.startswith('rgb') and f.endswith('.png') and f.split('-')[-1].split('.')[0] in ids]

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
# cfg.RPN_ANCHOR_SCALES = (10)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.TEST.EVAL_PERIOD = 100  # Validate the model every X iterations
cfg.SOLVER.MAX_ITER = 3500
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class 
cfg.INPUT.MASK_FORMAT = "bitmask" # To tell the trainer to not expect polygons
cfg.OUTPUT_DIR = output_dir

print("CHECKPOINT: Configurations loaded")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

print("CHECKPOINT: Model trained")

f = open(cfg.OUTPUT_DIR + '_config.yml', 'w')
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

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])

plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'train_val_metrics.png'))

print("CHECKPOINT: Data plotted")