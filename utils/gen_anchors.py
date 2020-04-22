import os 
import numpy as np 
from itertools import product
from numpy import sqrt
import matplotlib.pyplot as plt
import cv2
import torch

device = ('cuda' if torch.cuda.is_available() else 'cpu')
def get_anchors():
    """
    1. Paper uses the formula to calculate the scale of different boxes for each feature map.
       sk = s(min) + ((s(max) - s(min)) / (m-1)) * (k-1), where s(min) = 0.2 and s(max) = 0.9
    2. Also for 1 aspect ratio, he uses the scale, sk(new) = sqrt(sk * s(k+1))
    3. For the calculation of width and height of default boxes, width = sk * sqrt(aspect_ratio) 
       and height = sk / sqrt(aspect_ratio) 
    """
    feature_maps = {'x_1':14, 'x_2':7}
    scales = {'x_1':0.15, 'x_2':0.43}
    aspect_ratios = {'x_1':[1., 0.5], 'x_2':[1., 2., 3., 0.5, 2.5, 0.2]}
    # keys = list(feature_maps.keys())
    anchors = []
    for idx, key in enumerate(feature_maps):
        for i, j in product(range(feature_maps[key]), repeat = 2):
            center_x = (j + 0.5) / feature_maps[key] ## Adding 0.5 to get the center of pixel from (i,j)
            center_y = (i + 0.5) / feature_maps[key]
            for ratio in aspect_ratios[key]:
                width = scales[key] * sqrt(ratio)
                height = scales[key] / sqrt(ratio)
                anchors.append([center_x*224, center_y*224, width*224, height*224 ])

    anchors = torch.FloatTensor(anchors).to(device)  # (8732, 4)
    # anchors = anchors.clamp_(0, 224)  # (8732, 4)
    return anchors

if __name__ == "__main__":
    anchors = get_anchors()
    print(len(anchors))
