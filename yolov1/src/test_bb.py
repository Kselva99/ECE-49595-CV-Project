import cv2
import numpy as np


image = cv2.imread('../square_images/9ff3f191-7b8ffbb5.jpg')

with open('../annotation_txt/9ff3f191-7b8ffbb5.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    
    # Calculate box coordinates
    x1, y1 = int((x - w / 2) * image.shape[1]), int((y - h / 2) * image.shape[0])
    x2, y2 = int((x + w / 2) * image.shape[1]), int((y + h / 2) * image.shape[0])
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()