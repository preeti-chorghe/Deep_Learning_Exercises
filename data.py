from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data: pd.DataFrame , mode: str) -> None:
        super().__init__()

        # attributes
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean,train_std)])

        self.val_percentage = 0.2
        # split data in train and validation set
        #self.data_train = data.iloc[:int((1-val_percentage)*len(data))].reset_index()
        #self.data_validate = data.iloc[int((1-val_percentage)*len(data)):].reset_index()
        self.data =data
        self.mode = mode

    def __len__(self):
        if self.mode == "val":
            length = int(self.val_percentage*len(self.data))
        else:
            length = int((1-self.val_percentage)*len(self.data))
        return length
    
    def __getitem__(self, index):
        if self.mode == "val":
            data = self.data.iloc[:int((self.val_percentage)*len(self.data))].reset_index()
        else:
            data = self.data.iloc[int((self.val_percentage)*len(self.data)):].reset_index()

        # load image
        img_gray = imread(data["filename"][index])
        label = torch.tensor([data["crack"][index], data["inactive"][index]])

        # convert image to rgb
        img = gray2rgb(img_gray)

        # perform transforms
        if self._transform:
            img = self._transform(img)

        return img, label
