from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo # all models

import cv2
import numpy as np
import os

class Detector: 
    def __init__(self, model_type = "OD"):
        self.model_type = model_type
        print(f"Check point 2: Running model {model_type}")
        # Get configs
        self.cfg = get_cfg() 

        # Load model config and pretrained model
        # Full list of models found in detectron2/MODEL_ZOO.md
        if model_type == "OD":
            # Object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            # Instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP":
            # Keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS":
            # LVIS segmentation (for rare data types)
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "PS":
            # Panoptic segmentation (segmenting entire scene)
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        # Define detection thresholds
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" # cpu or cuda

        # Pass to default predictor
        self.predictor = DefaultPredictor(self.cfg)

    """Takes image path as input and outputs detection predictions on visualizer"""
    def onImage(self, imagePath, output_filename, color="SEGMENTATION"):
        # Read image using open-cv
        image = cv2.imread(imagePath)

        if self.model_type != "PS":
            # Get predictions
            predictions = self.predictor(image)

            # ColorMode options: IMAGE_BW, SEGMENTATION, IMAGE
            colormode = "ColorMode." + color
            
            # Initialize visualizer with image
            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = colormode) 

            # Draw predictions on input image using visualizer
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            # Get predictions
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

            # Initialize visualizer with image
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])) 

            # Draw predictions on input image using visualizer
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        print("Check point 3: Image processed")

        # # Display on screen (don't use to run on cluster, only locally)
        # cv2.imshow("Result", output.get_image()[:,:,::-1])
        # cv2.waitKey(0)

        # Directory where the image will be saved
        directory = "image-outputs"
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_path = os.path.join(directory, output_filename)
        print(f"Check point 4: Output path will be {full_path}")

        # Save image 
        cv2.imwrite(full_path, output.get_image()[:,:,::-1])
        print("Check point 5: Image outputted")