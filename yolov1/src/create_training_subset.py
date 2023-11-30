import os
import shutil
import random

def create_training_subset(source_dir, destination_dir, num_images):
    """
    Copies a random subset of images from the source directory to the destination directory.
    If the destination directory exists, it is deleted and recreated.

    :param source_dir: Path to the source directory containing images.
    :param destination_dir: Path to the destination directory where subset of images will be copied.
    :param num_images: Number of random images to be copied.
    """
    # Delete the destination directory if it exists
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)

    # Create the destination directory
    os.makedirs(destination_dir)

    # List all jpg files in the source directory
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # Randomly select a subset of images
    selected_images = random.sample(all_images, num_images)

    # Copy the selected images to the destination directory
    for image in selected_images:
        shutil.copy(os.path.join(source_dir, image), destination_dir)

    #print(f"Completed copying {num_images} images to {destination_dir}")
