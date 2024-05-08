from Detector import *
print("Check point 1: Import complete")

detector = Detector(model_type="IS")
detector.onImage("images/kitten.png", "plswork.png", "SEGMENTATION")

print("Complete")