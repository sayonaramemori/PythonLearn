from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import os

class MyDataset(Dataset):
    def __init__(self, class_one_dir, class_two_dir):
        # Prefix of the absolute path
        class_one_dir = class_one_dir if class_one_dir.endswith('/') else class_one_dir + '/'
        class_two_dir = class_two_dir if class_two_dir.endswith('/') else class_two_dir + '/'
        # All items in the specific directory and concat them with the Prefix
        self.data_items = []
        [self.data_items.append((class_one_dir + item,0)) for item in os.listdir(class_one_dir)]
        [self.data_items.append((class_two_dir + item,1)) for item in os.listdir(class_two_dir)]
    def __getitem__(self, index):
        path = self.data_items[index][0]
        item = Image.open(path)
        item_label = self.data_items[index][1]
        tensor = transforms.ToTensor()(item)
        #size = tensor.shape
        #print(f'Path is {path} with tensor size {size}')
        return tensor, item_label
    def __len__(self):
        return len(self.data_items)

    def check_img_size(self):
        res=set()
        for item_label in self.data_items:
            path, _ = item_label
            img= Image.open(path)
            tensor = transforms.ToTensor()(img)
            res.add(tensor.shape)
        [print(item) for item in res]
        return len(res)


if __name__ == "__main__":
    train_set = MyDataset("./0","./1")
    train_set.check_img_size()
    
