from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from torch.utils.data.dataset import TensorDataset

from pathlib import Path

from torch import from_numpy

import cv2

import os

import numpy as np

from PIL import Image,ImageFile
#solves truncated error message
ImageFile.LOAD_TRUNCATED_IMAGES = True

def our_loader_PIL(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, "rb") as f:
    #     img = Image.open(f)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.load()
    return img
    
def our_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

class UnlabelledImageFolder(DatasetFolder):
    """
    Normal dataset inherited of DatasetFolder from torchvision.
    Only real difference is the fact it doesnt return any labels, but two versions of the same image:
    1st image with a weak augmentation
    2nd image with a strong augmentation
    """
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = our_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        self.imgs = self.samples    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            weak_sample = self.transform(sample)
        if self.target_transform is not None:
            strong_sample = self.target_transform(sample)

        return weak_sample, strong_sample


class labelled_TensorDataset(TensorDataset):
    """
    Our dataset for Tensors, with the only difference of applying extra transformations to its
    image elements.
    This one is supposed to return an image and its label, for each __getitem()__ called.
    """
    def __init__(self,
                name: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                n_per_label : int = None):
        images = []
        labels = []

        for folder in ["nowildfire", "wildfire"]:
            for filename in os.listdir(os.path.join(name, folder)):
                image = cv2.imread(os.path.join(name, folder, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                images.append(image.transpose((2, 0, 1)))
                labels.append(1 if folder == "wildfire" else 0)

        images = from_numpy(np.asarray(images, dtype=np.uint8))
        labels = from_numpy(np.asarray(labels, dtype=np.uint8))
        
        super().__init__(images,labels)

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int):
        in_image = self.transform(self.tensors[0][index])
        #in_label = self.transform(self.tensors[1])
        in_label = self.tensors[1][index]
        #return tuple(self.transform(tensor[index]) for tensor in self.tensors)
        return (in_image.squeeze(),in_label.squeeze())


class unlabelled_TensorDataset(TensorDataset):
    """
    Our dataset for Tensors, with the only difference of applying extra transformations to its
    image elements.
    This one is supposed to return an image and its augmented version, for each __getitem()__ called. So it's made for being used with FixMatch or encoder training.
    """
    def __init__(self,
                name: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None, 
                include_labels: bool = False):
        images = []

        for folder in ["nowildfire", "wildfire"]:
            for filename in os.listdir(os.path.join(name, folder)):
                image = cv2.imread(os.path.join(name, folder, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image.transpose((2, 0, 1)))

                
        images = from_numpy(np.asarray(images, dtype=np.uint8))
        
        super().__init__(images)

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index: int):
        in_image = self.transform(self.tensors[0][index])
        in_augmented_image = self.target_transform(self.tensors[0][index])

        return (in_image.squeeze(),in_augmented_image.squeeze())



