"""
This is a backup script in case plotting doesn't work. It takes any metrics.json file and plots its total_loss
"""

import os
import json
import argparse
import matplotlib.pyplot as plt

# Handle CLAs
parser = argparse.ArgumentParser(description="Plot metrics.json.")

parser.add_argument('-m', '--metrics', 
                    type=str, 
                    required=True, 
                    help="The file where metrics are stored.")

parser.add_argument('-o', '--output_dir', 
                    type=str, 
                    required=True, 
                    help="The directory where the outputs are stored.")

parser.add_argument('-n', '--name', 
                    type=str, 
                    required=True, 
                    help="The name of the plot.")

args = parser.parse_args()

metrics_path = os.path.abspath(args.metrics)

# Error handling
if not os.path.isdir(metrics_path):
    print(f"Error: The specified file does not exist")
    exit(0)

# Load metrics
with open(metrics_path) as f:
    metrics = [json.loads(line) for line in f]

# Extract the metrics
iterations = [x["iteration"] for x in metrics]
losses = [x["total_loss"] for x in metrics]
# You can add more metrics as needed

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(iterations, losses, label='Loss')
# Add more plots as needed
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.abspath(args.output_dir + args.name))
