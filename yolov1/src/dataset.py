from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

class YoloDataset(Dataset):
    def __init__(self, data_dict, image_folder, transform=None):
        self.data_dict = data_dict
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        image_name = list(self.data_dict.keys())[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        target = self.data_dict[image_name]
        if self.transform:
            image = self.transform(image)
        return image, target






# Example usage:

# image_folder = '/path/to/your/images'
# transform = transforms.Compose([
#     transforms.Resize((448, 448)),
#     transforms.ToTensor(),
# ])


#dataset = YOLODataset(tensor_dict, image_folder, transform=transform)
#dataloader = DataLoader(dataset, batch_size=4, shuffle=True)









# sample code
#dataset = YoloDataset(tensor_dict, image_folder, transform=transform)
#dataloader = DataLoader(dataset, batch_size = 8, shuffle = False)












# what is tensor dict hold?


#tensor_dict
# key: image dataset
# value: tensor that holds the output neurons labels


# need to use dataset class to store image and tensors, to send to dataloader.







