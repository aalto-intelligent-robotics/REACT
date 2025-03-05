import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", type=str, default='./instance_views')
opts = parser.parse_args()
# Configuration
source_dir = opts.source_dir  # Replace with the path to your source directory
output_dir = 'dataset'  # Replace with the desired output directory
train_ratio = 0.8  # Adjust the ratio as needed

# Ensure output directories exist
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through each subdirectory in the source directory
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        # List all image files in the subdirectory
        images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        
        # Shuffle images randomly
        random.shuffle(images)

        # Split images into train and validation sets
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create subdirectories in the train and val folders
        train_subdir = os.path.join(train_dir, subdir)
        val_subdir = os.path.join(val_dir, subdir)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)

        # Copy images to the respective directories
        for img in train_images:
            shutil.copy2(os.path.join(subdir_path, img), os.path.join(train_subdir, img))

        for img in val_images:
            shutil.copy2(os.path.join(subdir_path, img), os.path.join(val_subdir, img))

print("Dataset split into train and val successfully.")
