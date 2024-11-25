#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAI implementation of the paper:
'Deep supervised hashing for fast retrieval of radio image cubes'
Original author: Steven Machetho (s.n.machetho@rug.nl)   
"""


import os
import shutil
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fastai.data.all import *
from fastai.vision.all import *
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


seed = 42
hyper_params = {
    'epochs': 200,
    'lr': 1e-3,
    'bits': 8,
    'dataset': '../../data'
}

data_path = Path(hyper_params["dataset"])
output_dir = './output/'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

# set parameters for reproducibility
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# pre-process and load the data
data_block = DataBlock(
    blocks = (ImageBlock,CategoryBlock),
    get_items = get_image_files,
    splitter = GrandparentSplitter(),
    item_tfms=Resize(224),
    get_y = parent_label)

data = data_block.dataloaders(data_path)

# define model
class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hd = nn.Sequential(AdaptiveConcatPool2d(),
                                Flatten(),
                                nn.BatchNorm1d(in_features),
                                nn.Dropout(p = 0.5),
                                nn.Linear(in_features, 512),
                                nn.ReLU(inplace = True),
                                nn.BatchNorm1d(512),
                                nn.Dropout(p = 0.2),
                                nn.Linear(512, out_features),
                                nn.Sigmoid())
        
    def forward(self, x):
        return self.hd(x)

loss_function = losses.TripletMarginLoss(margin=0.2,
                                         distance = CosineSimilarity(),
                                         reducer = ThresholdReducer(high=0.3), 
                                         embedding_regularizer = LpRegularizer())

model = vision_learner(data,
                       models.densenet161,
                       custom_head = CustomHead(4416, hyper_params["bits"]),
                       loss_func = loss_function,
                       metrics = error_rate)

# freeze the earlier layers and keep only the last layer trainable
model.freeze()

# train the custom head 
model.fit_one_cycle(1)

# unfreeze the model
model.unfreeze()

# train the model
model.fit_one_cycle(n_epoch = hyper_params["epochs"], lr_max=slice(hyper_params["lr"]),wd=1e-3)
model.recorder.plot_loss()
plt.savefig(output_dir + 'Densenet_Train_and_Test_Loss_curve_at_freeze.png')

# save the model
model.save("densenet_model_weights")
# model.export('densenet161.pkl')
