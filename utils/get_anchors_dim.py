import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class Uplara(Dataset):
    def __init__(self, dataset_path, image_path, transform = None, image_size = 224, train_from_augmentation = False, augmentation = False): ##Augmentation enables the augmentation.py to use Dataset
        self.dataset = pd.read_csv(dataset_path)
        self.foot_id = self.dataset['foot_id']
        # self.image_url = self.dataset['url']
        self.transform  = transform
        self.image_size = image_size
        self.augmentation = augmentation
        self.train_from_augmentation = train_from_augmentation
        self.image_path = image_path
    def __getitem__(self, idx):
        # print(self.foot_id[idx])
        # image = Image.open(urllib.request.urlopen(self.image_url[idx]))  #when scrapping images directly
        image_path = self.image_path + str(self.foot_id[idx]) +"_" + str(self.dataset['angle'][idx]) +".jpg"  #Using already scrapped images
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        image = image.convert('RGB')
        image = np.array(image)

        ###################For Left Foot#######################
        l_xmin = np.min(self.dataset.loc[idx][::2][1:26])
        l_ymin = np.min(self.dataset.loc[idx][1:][::2][1:26])
        l_xmax = np.max(self.dataset.loc[idx][::2][1:26])
        l_ymax = np.max(self.dataset.loc[idx][1:][::2][1:26])
        # cv2.circle(image, (int(l_xmin), int(l_ymin)),2,  (0,255,0), 2)
        # cv2.circle(image, (int(l_xmax), int(l_ymax)),2,  (0,255,0), 2)
        #transformation of coordinates
        l_prob = self.dataset.loc[idx][-2]
        ###################For Right Foot #######################
        r_xmin = np.min(self.dataset.loc[idx][52:][::2][0:25])
        r_ymin = np.min(self.dataset.loc[idx][51:][::2][1:26])
        r_xmax = np.max(self.dataset.loc[idx][52:][::2][0:25])
        r_ymax = np.max(self.dataset.loc[idx][51:][::2][1:26])
        # cv2.circle(image, (int(r_xmin), int(r_ymin)),2,  (0,255,0), 2)
        # cv2.circle(image, (int(r_xmax), int(r_ymax)),2,  (0,255,0), 2)
        # plt.imshow(image)
        # plt.show()
        #transformation of coordinates
        r_prob =self.dataset.loc[idx][-1]

        if (self.image_size == 512):
            ratio = self.image_size / 512
            l_xmin, l_ymin, l_xmax, l_ymax = l_xmin*ratio , l_ymin*ratio , l_xmax*ratio ,l_ymax*ratio 
            r_xmin, r_ymin, r_xmax, r_ymax =  r_xmin * ratio, r_ymin * ratio, r_xmax * ratio, r_ymax * ratio
        l_box = [l_xmin, l_ymin, l_xmax, l_ymax]
        r_box = [r_xmin, r_ymin, r_xmax, r_ymax]
        boxes = [l_box, r_box]
        labels = [l_prob, r_prob]

        if self.augmentation:
            return image, boxes, labels, self.foot_id[idx]
        ###################### Visualize the bounding boxes ##############################
        # for visualizing the bounding boxes use, augmentation.py file

        target, label = self.encoder(boxes, labels)  # To get the encoding of the target

        
        return image, target, label

    def encoder(self, boxes, labels):

        targets = []
        label = []
        # for left box
        if not (labels[0] == 0):
            target = np.zeros(( 4))
            xmin, ymin, xmax, ymax = boxes[0]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            target[0:4] = center_x, center_y, width, height
            label.append(labels[0])
            targets.append(target)
        # for right box
        if not (labels[1] == 0):
            target = np.zeros(( 4))
            xmin, ymin, xmax, ymax = boxes[1]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            target[0:4] = center_x, center_y, width, height
            label.append(labels[1])
            targets.append(target)
            # print(center_x, center_y, width, height)
        return targets, label

    def __len__(self):
        return len(self.foot_id)
    
   


if __name__ == "__main__":
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_dataset.csv","/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_images/", )
    for i in range(len(dataset)):
        image, boxes, label = dataset[i]
        boxes = boxes[0]
        print(boxes)
        xmin = int(boxes[0] - boxes[2] / 2)
        ymin = int(boxes[1] - boxes[3] / 2)
        xmax = int(boxes[0] + boxes[2] / 2)
        ymax = int(boxes[1] + boxes[3] / 2)
        aspect_ratio = boxes[2] / boxes[3]
        area = (boxes[2] * boxes[3])
        scale = area / (224*224)
        scale = np.sqrt(scale)
        print(aspect_ratio  , scale)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255, 0), 1)
        plt.imshow(image)
        plt.show()