import os
from create_training_subset import create_training_subset
from images_yolo_format import images_yolo_format
from create_labels_txt import create_yolo_labels
from gen_labels import create_tensors_for_folder
from dataset import YoloDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from yolov1loss import Yolov1Loss
from yolov1 import YOLOv1
from train_model import train_model
import torch
import torch.nn as nn

def main():
    # Define paths
    training_set_original = "../../bdd100k/train"
    training_set = "../train_set"
    yolo_training_set = "../resized_set"
    scale_pad_file = "../scale_pad_info.json"
    json_path ="../../bdd100k/labels/bdd100k_labels_images_train.json"
    annotation_txt = "../annotation_txt"


    # can comment out steps 1-3 if already generated.

    # Step 0: Create training subset
    num_training_images = int(input("Enter the number of images for the training subset: "))
    
    create_training_subset(training_set_original, training_set, num_training_images)
    print("Step 0 Done: Created training subset consisting of " + str(num_training_images) + " images.\n ")

    

    # Step 1: Resize and pad images
    images_yolo_format(training_set, yolo_training_set, scale_pad_file)
    print("Step 1 Done: Converted 1280x720 images into 448x448 images.\n ")


    # Step 2: Create YOLO labels from JSON
    create_yolo_labels(json_path, yolo_training_set, annotation_txt, scale_pad_file)
    print("Step 2 Done: Created txt files of bounding box data in YOLO format. \n")



    # Step 3: Generate tensor labels for each image
    tensor_dict = create_tensors_for_folder(annotation_txt, S=7, B=2, C=3, car_class_id=2)
    print("Step 3 Done: Generated label tensors for training. \n")

    # printing a sample output tensor
    #print(tensor_dict['0a0a0b1a-7c39d841.jpg']) 



    # Step 4: Create Dataset and DataLoader
    # Define transformations (assuming images are already resized to 448x448)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations you need
    ])

    batch_size = int(input("Enter batch size: "))    
    dataset = YoloDataset(tensor_dict, yolo_training_set, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Step 4 Done: Setup Dataloader. \n")




    # Training?
    # Device configuration - use GPU if available
    print("Setting up Training. \n")
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU. \n")
        torch.cuda.empty_cache()
    else:
        raise Exception


    device = torch.device('cuda')
    model = YOLOv1(grid_size=7, num_bboxes=2, num_classes=3).to(device)
    criterion = Yolov1Loss(S=7, B=2, C=3).to(device)
    lr = float(input("Enter learning rate:  "))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = int(input("Enter number of epochs:  "))



    print("Begin Training \n")
    # Call the training function
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

    torch.save(trained_model.state_dict(), 'trained_yolov1_model.pth')


if __name__ == "__main__":
    main()

