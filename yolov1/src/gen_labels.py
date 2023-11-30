import os
import torch


# Creates labels (output neuron values) based on bounding box and image class
# Does it for all images in a folder



# This code has 3 functions:
# 1. read_label_file
# reads txt data and produces list of bounding box/class info to process
# format: class_id, x_center, y_center, width, height


# 2. process_labels_with_priority
# responsible for creating a label tensor for a single image from txt data. 

# 3. create_tensors_for_folder
# builds dictionary of tensors representing
# key: image_name.jpg, value: image_label_tensor


def process_labels_with_priority(labels, S=7, B=2, C=20, car_class_id=2):
    """
    Process YOLO format labels for a single image into a target tensor,
    prioritizing car labels.

    Parameters:
    - labels: List of label data, each containing class_id, x_center, y_center, width, height
    - S: Size of the grid (e.g., 7 for YOLOv1).
    - B: Number of bounding boxes per cell (e.g., 2 for YOLOv1).
    - C: Number of classes.
    - car_class_id: The class ID for cars.

    Returns:
    - target: A tensor of shape (S, S, B*5+C) representing target labels.
    """
    target = torch.zeros((S, S, B*5 + C))
    for label in labels:
        class_id, x, y, w, h = label
        grid_x = int(x * S)
        grid_y = int(y * S)

        # If the grid cell is already full, prioritize car labels
        if target[grid_y, grid_x, C:].sum() >= 2 * B:
            if class_id == car_class_id:
                # Look for a non-car bounding box to replace
                for b in range(B):
                    bbox_start = C + b*5
                    if target[grid_y, grid_x, bbox_start:bbox_start+5].sum() != 0 and \
                       target[grid_y, grid_x, bbox_start:bbox_start+5][4] != car_class_id:
                        # Replace the existing non-car bounding box
                        target[grid_y, grid_x, bbox_start:bbox_start+5] = torch.tensor([x*S % 1, y*S % 1, w*S, h*S, 1])
                        break
            continue

        # Find an available bounding box slot
        for b in range(B):
            bbox_start = C + b*5
            if target[grid_y, grid_x, bbox_start:bbox_start+5].sum() == 0:
                # Assign the new bounding box
                target[grid_y, grid_x, bbox_start:bbox_start+5] = torch.tensor([x*S % 1, y*S % 1, w*S, h*S, 1])
                target[grid_y, grid_x, class_id] = 1  # Set the class label
                break

    return target






def read_label_file(label_path, S=7, B=2, C=3, car_class_id=2):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        class_id, x, y, w, h = [float(x) if i else int(x) for i, x in enumerate(line.strip().split())]
        labels.append((class_id, x, y, w, h))
    return labels



def create_tensors_for_folder(folder_path, S=7, B=2, C=3, car_class_id=2):
    tensor_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            image_name = file_name.replace('.txt', '.jpg')
            label_path = os.path.join(folder_path, file_name)
            labels = read_label_file(label_path, S=S, B=B, C=C, car_class_id=car_class_id)
            target_tensor = process_labels_with_priority(labels, S=S, B=B, C=C, car_class_id=car_class_id)
            tensor_dict[image_name] = target_tensor
    return tensor_dict

# Usage example
#folder_path = os.path.expanduser('~/495/annotation_txt/')  # Replace with your actual folder path
#tensor_dict = create_tensors_for_folder(folder_path, S=7, B=2, C=3, car_class_id=2)


#print(tensor_dict)
