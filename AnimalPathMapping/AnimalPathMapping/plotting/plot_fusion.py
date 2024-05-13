"""
This is a manual plotting script
"""

import os
import json
import argparse
import matplotlib.pyplot as plt

x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y = [0.189, 0.350, 0.358, 0.318, 0.292, 0.242, 0.227, 0.165, 0.087, 0.099, 0.086]
y2 = [0.319, 0.539, 0.510, 0.499, 0.465, 0.461, 0.439, 0.422, 0.394, 0.381, 0.376]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.xlabel('Percentage of Thermal Imagery')
plt.ylabel('Average Precision')
plt.title('Average Precision vs. Percentage of Thermal Imagery (Separated Masks, IOU=0.5)')
plt.grid(True)

# Save the plot
plt.savefig('./experiment_4/sep_test_metrics.png')
