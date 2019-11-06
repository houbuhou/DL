import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from matplotlib import pyplot as plt
from PIL import Image
import random


class EPI(data.Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.files = os.listdir(self.image_paths)
        check = 0
        while (check < len(self.files)):
            if (self.files[check].find('.tif') < 0):
                remove_ele = self.files[check]
                # print(self.files[check])
                self.files.remove(remove_ele)
                check -= 1
            check += 1
        #         print(self.files)
        self.file_name = (f[:-4] for f in self.files)
        self.file_name = list(self.file_name)
        # print(self.file_name)
        self.mask_name = os.listdir(self.target_paths)
        self.labels = os.listdir(self.target_paths)

    def pre_process(self, image):
        raw_image = np.array(image, dtype=np.uint8)
        HSV_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        H = HSV_image[:, :, 0]
        #         S = cv2.equalizeHist(HSV_image[:,:,1])
        S = HSV_image[:, :, 1]
        V = cv2.equalizeHist(HSV_image[:, :, 2])
        H = np.expand_dims(H, axis=2)
        S = np.expand_dims(S, axis=2)
        V = np.expand_dims(V, axis=2)
        HSV_eq = np.concatenate([H, S, V], axis=2)
        eq = cv2.cvtColor(HSV_eq, cv2.COLOR_HSV2BGR)
        eq = Image.fromarray(np.uint8(eq))
        #         cv2.imshow("eq",eq)
        #         cv2.waitKey()
        return eq

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(1000, 1000))
        image = resize(image)
        mask = resize(mask)

        # colorjitter
        trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.25, saturation=0.25, contrast=0.25)
        ])
        if random.random() > 0.5:
            image = trans(image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths + self.files[index])
        mask_name = str(self.mask_name)
        # print(mask_name.find(self.file_name[index]))
        if (mask_name.find(self.file_name[index]) < 0):
            mask = np.zeros([1000, 1000], dtype=np.uint8)
        else:
            mask = Image.open(self.target_paths + self.file_name[index] + '_mask.png')
        image = Image.fromarray(np.uint8(image))
        mask = Image.fromarray(np.uint8(mask))
        # plt.figure(0)
        # plt.imshow(image)
        # plt.figure(1)
        # plt.imshow(mask)
        # plt.show()
        image_pro = self.pre_process(image)
        x, y = self.transform(image_pro, mask)
        return x, y

    #         return image, mask

    def __len__(self):
        return len(self.file_name)