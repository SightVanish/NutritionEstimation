import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import cv2

class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []

        for line in lines_rgb:
            image_rgb = line.split()[0]
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass =  line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

        self.transform = transform

    #RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])  
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB))
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd,cv2.COLOR_BGR2RGB))
        except Exception as e:
            print("IMAGE: ", self.images[index], " Error: ", e)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        # return 2 images
        return img_rgb, \
            self.labels[index], \
            self.total_calories[index], \
            self.total_mass[index], \
            self.total_fat[index], \
            self.total_carb[index], \
            self.total_protein[index], \
            img_rgbd

    def __len__(self):
        return len(self.images)
