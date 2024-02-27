import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import PIL 
from PIL import Image
import torchvision.transforms as transforms
'''
class Personalize0(Dataset):
    def __init__(self,
                 txt_file,
                 size=128,
                 interpolation="bicubic",
                 flip_p=0.15,
                 is_mri=False,
                 val=False
                 ):
        self.data_paths = txt_file

        self.image_paths=pd.read_csv(self.data_paths, header=None)[0]

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [l for l in self.image_paths],
        }
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(self.image_paths[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

'''

class Personalize0(Dataset):
    def __init__(self,
                 txt_file,
                 size=128,
                 interpolation="bicubic",
                 flip_p=0.15
                 ):
        self.data_paths = txt_file

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [l for l in self.image_paths],
        }
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    
class PersonalizeTrain0(Personalize0):
    def __init__(self, csv_path_train, **kwargs):
        super().__init__(txt_file=csv_path_train)

class PersonalizeVal0(Personalize0):
    def __init__(self, csv_path_val, flip_p=0., **kwargs):
        super().__init__(txt_file=csv_path_val ,
                         flip_p=flip_p)