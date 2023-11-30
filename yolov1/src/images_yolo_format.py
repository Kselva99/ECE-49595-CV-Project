import cv2
import numpy as np
import os
import json
import shutil

def resize_and_pad_image(image, target_size=448):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad

    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    
    # Return image, scale factors, and padding offsets
    return padded_image, (scale, scale), (left_pad, top_pad)

def images_yolo_format(folder_path, output_folder_path, scale_pad_file):
    scale_pad_info = {}

    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    
    os.makedirs(output_folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            
            if image is not None:
                processed_image, scale_factor, pad_offset = resize_and_pad_image(image)
                cv2.imwrite(os.path.join(output_folder_path, filename), processed_image)
                scale_pad_info[filename] = {"scale": scale_factor, "pad": pad_offset}
            else:
                print("Failed to load image {filename}")

    # Save scale and padding information
    with open(scale_pad_file, 'w') as file:
        json.dump(scale_pad_info, file)

