"""
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)

Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run this script from the miniImagenet directory:
    cd data/miniImagenet/
    python proc_images.py
"""

from __future__ import print_function
import csv
import glob
import os
from PIL import Image

# Path to the images folder
path_to_images = 'images/'

# Recursively find all images in subdirectories
all_images = glob.glob(path_to_images + '**/*.JPEG', recursive=True)

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((84, 84), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(f"Resized {i} images")

# Put images in the correct directory based on the CSV files
for datatype in ['train', 'val', 'test']:
    # Create the main directory for the datatype (train, val, test)
    os.makedirs(datatype, exist_ok=True)

    # Open the CSV file for the datatype
    with open(datatype + '.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            image_path = row[0]  # First column is the image path (e.g., n01532829/n01532829_721.JPEG)
            label_name = row[1]  # Second column is the label name

            # Construct the source path (relative to the images folder)
            source_path = os.path.join(path_to_images, image_path)

            # Construct the target directory and path
            target_dir = os.path.join(datatype, label_name)
            os.makedirs(target_dir, exist_ok=True)  # Create the label directory if it doesn't exist
            target_path = os.path.join(target_dir, os.path.basename(image_path))

            # Move the image to the target directory
            if os.path.exists(source_path):
                os.rename(source_path, target_path)
            else:
                print(f"Warning: {source_path} does not exist. Skipping.")