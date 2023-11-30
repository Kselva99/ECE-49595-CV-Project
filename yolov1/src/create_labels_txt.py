import json
import os
import shutil

def adjust_bounding_box(box, scale_factor, pad_offset, img_width=448, img_height=448):
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    x1_scaled = x1 * scale_factor[0] + pad_offset[0]
    y1_scaled = y1 * scale_factor[1] + pad_offset[1]
    x2_scaled = x2 * scale_factor[0] + pad_offset[0]
    y2_scaled = y2 * scale_factor[1] + pad_offset[1]

    x_center = ((x1_scaled + x2_scaled) / 2) / img_width
    y_center = ((y1_scaled + y2_scaled) / 2) / img_height
    width = (x2_scaled - x1_scaled) / img_width
    height = (y2_scaled - y1_scaled) / img_height
    return x_center, y_center, width, height

def create_yolo_labels(json_path, img_folder, output_folder, scale_pad_file):
    with open(json_path, 'r') as file:
        data = json.load(file)

    with open(scale_pad_file, 'r') as file:
        scale_pad_info = json.load(file)

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    
    os.makedirs(output_folder)

    category_mapping = {'traffic light': 0, 'traffic sign': 1, 'car': 2} # extend as needed

    for item in data:
        image_name = item['name']
        if image_name in os.listdir(img_folder):
            output_path = os.path.join(output_folder, image_name.replace('.jpg', '.txt'))
            with open(output_path, 'w') as f:
                for label in item.get('labels', []):
                    if 'box2d' in label and image_name in scale_pad_info:
                        scale_factor = scale_pad_info[image_name]["scale"]
                        pad_offset = scale_pad_info[image_name]["pad"]
                        yolo_box = adjust_bounding_box(label['box2d'], scale_factor, pad_offset)
                        class_id = category_mapping.get(label['category'], -1)
                        if class_id != -1:
                            f.write("{} {}\n".format(class_id, ' '.join(map(str, yolo_box))))




# json_path = os.path.expanduser('~/495/bdd100k_labels_images_train.json')
# img_folder = os.path.expanduser('~/495/square_images')
# output_folder = os.path.expanduser('~/495/annotation_txt')
# scale_pad_file = os.path.expanduser('~/495/scale_pad_info.json')

# create_yolo_labels(json_path, img_folder, output_folder, scale_pad_file)
