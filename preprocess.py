import cv2
import numpy as np
import os
import sys
import subprocess

def load_image(image_path):
    image = cv2.imread(image_path)

    return image

def preprocess(image, model_size=(416, 416)):
    h, w = image.shape[:2]
    mh, mw = model_size

    scale = min(mw/w, mh/h)
    nh, nw = int(scale * h), int(scale * w)

    image_pad = np.full((mh, mw, 3), 128)
    dw, dh = (mw - nw) // 2, (mh - nh) // 2
    image_pad[dh:nh+dh, dw:nw+dw, :] = cv2.resize(image, (nw, nh))

    image_norm = image_pad / 255.0

    return image_norm

def main(model_version, dataset_split):
    if (model_version) == 'yolov1':
        v1_path = 'insert later'
        subprocess.run(['python', v1_path, dataset_split], check=True)
    else:
        data_dir = f"../bdd100k/{dataset_split}" #fix later
        output_dir = f"../preprocessed_{model_version}/{dataset_split}"

        os.makedirs(output_dir, exist_ok=True)

        for image in os.listdir(data_dir):
            image_path = os.path.join(data_dir)
            image = cv2.imread(image_path)
            preprocessed_img = preprocess(image, model_version)


            output_path = os.path.join(output_dir, image)
            cv2.imwrite(output_path, preprocessed_img * 255)