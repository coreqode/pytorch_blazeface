import torch
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from PIL import Image
from tqdm import tqdm, tnrange, tqdm_notebook
import math
from collections import defaultdict
from model.BlazeFace import BlazeFace
from utils.dataset import Uplara
from utils.multibox_loss import MultiBoxLoss
from utils.gen_anchors import get_anchors
from utils.earlystopping import EarlyStopping
from utils.RAdam import RAdam
from utils.LookAhead import Lookahead

### Checking for GPU-----------------------------------------------------------------------------------------
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = ('cuda' if torch.cuda.is_available() else 'cpu')
### Preprocessing--------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_dataset.csv", "/home/noldsoul/Desktop/Uplara/dataset/augmented_images/augmented_images/", transform = transform)
# print(len(dataset))
trainset, valset = random_split(dataset,[400, 27])
train_loader = DataLoader(trainset, batch_size = 8, shuffle = True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(valset, batch_size =8, shuffle = False,collate_fn=dataset.collate_fn)

# ##Initialize the Model
model = BlazeFace()
model = model.to(device)
# # model.load_state_dict(torch.load('', map_location=torch.device(device)))
criterion = MultiBoxLoss(priors_cxcy=get_anchors()).to(device)
# image, boxes, label= next(iter(train_loader))
# image = image.to(device)
# output = model(image)
# loss = criterion( output[1], output[0], boxes,  label)
# loss.backward()

## Training the model--------------------------------------------------------------------------------
n_epochs = 150
patience = 5 #used for early stopping
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
optimizer = RAdam( model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, degenerated_to_sgd=True) #Rectified Adam
# optimizer =  Lookahead(base_optimizer,1e-3 ,k = 6)
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0.005, diff =0.05)
valid_loss_min = np.Inf
epoch_tqdm = tqdm(total = n_epochs, desc = 'epochs')
for epoch in range(n_epochs):
    train_tqdm = tqdm(total = len(train_loader), desc = 'training batch')
    ###################
    # train the model #
    ###################
    model.train()
    for batch_idx, (image, boxes,label) in enumerate(train_loader):
        if train_on_gpu:
            image = image.cuda()
            model = model.cuda()
        optimizer.zero_grad()
        output = model.forward(image)
        loss = criterion( output[1], output[0], boxes,  label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_tqdm.update(1)
    train_tqdm.close()
    ######################    
    # validate the model #
    ######################
    valid_tqdm = tqdm(total = len(val_loader), desc ='validation batch')
    model.eval()
    threshold = []
    for batch_idx,(image, boxes, label) in enumerate(val_loader):
        if train_on_gpu:
            image = image.cuda()
        output = model(image)
        loss = criterion( output[1], output[0], boxes,  label)
        val_losses.append(loss.item())
        valid_tqdm.update(1)
    valid_tqdm.close()
    train_loss = np.average(train_losses)
    val_loss = np.average(val_losses)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_loss, val_loss,))
    train_losses = []
    val_losses = []
    early_stopping(val_loss, model, train_loss)
    if early_stopping.early_stop:
      print('Early Stopping')
      break
    epoch_tqdm.update(1)



