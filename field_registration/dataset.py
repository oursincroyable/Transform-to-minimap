import os
import random

import numpy as np
from PIL import Image
import skimage.segmentation as seg

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class SemanticSegmentationDataset(Dataset):
    """
    This class defines a custom dataset containing Worldcup or TS-Worldcup for keypoints detection
    """

    def __init__(self, root_dir, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """

        self.root_dir = root_dir
        self.train = train

        self.images = [] #list of paths of images
        self.annotations = [] #list of paths of homography matrices

        self.img_dir = os.path.join(self.root_dir, 'Dataset', '80_95')
        self.ann_dir = os.path.join(self.root_dir, 'Annotations', '80_95')

        PATH = os.path.join(self.root_dir, 'train' if train else 'test')

        with open(PATH + '.txt', 'r') as lines:
            for line in lines:
                video = line.rstrip('\n')
                for img in sorted(os.listdir(os.path.join(self.img_dir, video))):
                    self.images.append(os.path.join(self.img_dir, video, img))
                for hom in sorted(os.listdir(os.path.join(self.ann_dir, video))):
                    self.annotations.append(os.path.join(self.ann_dir, video, hom))

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __fromhomography(self, mat):
        '''
        Make a groundtruth segmentation mask describing projected keypoints in the image from a homography matrix

        Args:
            mat : 3*3 numpy array representing the homography matrix

        Return:
            label : 720*1280 numpy array representing the segmentation mask, 0 for background, i for ith keypoints
        '''
        field_width = 114.83
        field_height = 74.37

        dw = np.linspace(0, field_width, 13)
        dh = np.linspace(0, field_height, 7)

        grid_dw, grid_dh = np.meshgrid(dw, dh, indexing='ij')
        grid_field = np.stack((grid_dw, grid_dh), axis=2).reshape(-1, 2)

        proj_grid = np.concatenate((grid_field, np.ones((91, 1))), axis=1) @ np.linalg.inv(mat.T)
        proj_grid /= proj_grid[:, 2, np.newaxis]

        for i in range(0, len(proj_grid)):
            proj_grid[i][2] = i + 1

        # proj_img_grid = keypoints in the projected field on the image
        proj_img_grid = []
        for p in proj_grid:
            # image resolution = 1280*720
            if 0 <= p[0] < 1280 and 0 <= p[1] < 720:
                proj_img_grid.append(p)

        # produce the groundtruch label in full resolution

        label = np.zeros((720, 1280), dtype=np.float32)

        for p in proj_img_grid:
            x = np.rint(p[0]).astype(np.int32)
            y = np.rint(p[1]).astype(np.int32)

            if 0 <= x < 1280 and 0 <= y < 720:
                label[y, x] = p[2]

        label = seg.expand_labels(label, distance=20)

        return label

    def __randomhflip(self, image, hom):
        '''
        Apply random horizontal flip to a pair of an image and a homography matrix

        Args:
            image: 3*720*1280 tensor of football broadcast image
            hom: 3*3 numpy array representing a homography matrix

        Return:
            image: 3*720*1280 tensor of football broadcast image
            label: 720*1280 numpy array representing the segmentation mask
        '''
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            hom = np.array([[-1, 0, 114.83],
                            [0, 1, 0],
                            [0, 0, 1]]) @ hom @ np.array([[-1, 0, 1280],
                                                          [0, 1, 0],
                                                          [0, 0, 1]])
            label = self.__fromhomography(hom)
            label = transforms.ToTensor()(label)
            return image, label

        else:
            label = self.__fromhomography(hom)
            label = transforms.ToTensor()(label)

            return image, label

    def __randomcrop(self, image, label):
        '''
        Apply random crop to a pair of an image and a segmentation mask
        '''
        if random.random() > 0.5:
            scale = random.uniform(0.7, 1)
            h = round(720 * scale)
            w = round(1280 * scale)

            tmp = transforms.RandomCrop((h, w))(torch.concat((image, label), dim=0))
            image = tmp[0:3, :, :]
            label = torch.squeeze(tmp[3, :, :], dim=0)
            image = transforms.Resize(size=(720, 1280), interpolation=transforms.InterpolationMode.BICUBIC)(image)
            label = transforms.Resize(size=(720, 1280), interpolation=transforms.InterpolationMode.NEAREST)(
                torch.unsqueeze(label, dim=0))
        return image, torch.squeeze(label, dim=0)

    def __randomerase(self, image):
        '''
        Apply a random erase to an image
        '''
        if random.random() > 0.5:
            for _ in range(20):
                image = transforms.RandomErasing(p=0.5, scale=(0.0008, 0.001), ratio=(100 / 45, 101 / 45), value=127.5)(
                    image)
        return image



    def __getitem__(self, idx):

        image = Image.open(self.images[idx])
        hom = np.load(self.annotations[idx])

        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        if self.train:
            # image augmentation
            image = self.__randomerase(image)
            image, label = self.__randomhflip(image, hom)
            image, label = self.__randomcrop(image, label)

        else:
            label = self.__fromhomography(hom)
            label = transforms.ToTensor()(label)

        # image, label resize
        image = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.BICUBIC)(image)
        label = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.NEAREST)(
            torch.unsqueeze(label, dim=0))

        data = {"pixel_values": image, "labels": torch.squeeze(label, dim=0), "image_path": self.images[idx]}

        return data
