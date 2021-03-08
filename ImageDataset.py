import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name).convert("RGB") 
        height = I.size[0]
        width = I.size[1]
        if height < 384 or width<384:
            if height<width:
                I = I.resize((384, int(width*384/height)), Image.BICUBIC)
            else:
                I = I.resize((int(height*384/width), 384), Image.BICUBIC)
    return I
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        if test == False: # train
            self.data = pd.read_csv(csv_file, sep=',', header=None)
        else: # test
            self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)

            mos = self.data.iloc[index, 1]
            # std = self.data.iloc[index, 2]
            sample = {'I': I, 'mos': mos}
        else:
            image_name1 = os.path.join(self.img_dir, self.data.iloc[index, 1])
            image_name2 = os.path.join(self.img_dir, self.data.iloc[index, 2])
            image_name3 = os.path.join(self.img_dir, self.data.iloc[index, 3])
            image_name4 = os.path.join(self.img_dir, self.data.iloc[index, 4])

            # print(image_name1)
            I1 = self.loader(image_name1)
            I2 = self.loader(image_name2)
            I3 = self.loader(image_name3)
            I4 = self.loader(image_name4)

            if self.transform is not None:
                I1 = self.transform(I1)
                I2 = self.transform(I2)
                I3 = self.transform(I3)
                I4 = self.transform(I4)
            
            y = torch.FloatTensor(self.data.iloc[index, 5:11].tolist())
            v = self.data.iloc[index, 11]
            sample = {'I1':I1, 'I2':I2, 'I3':I3, 'I4':I4, 'y': y, 'v':v ,'idx': index}
        return sample

    def __len__(self):
        return len(self.data.index)
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from Transformers import AdaptiveResize
    train_transform = transforms.Compose([
            #transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    train_data = ImageDataset(csv_file='../gen_img/new_train_pair_0.07_1_thers0.5_binary.txt',
                               img_dir='../gen_img',
                               transform= train_transform,
                               test=False)
    train_loader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8)
    for step, sample_batched in enumerate(train_loader, 0):
        print(sample_batched['y'])
        print(sample_batched['y_binary'])
