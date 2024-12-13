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
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


dic_labels = { 
    'FR1':0,
    'FR2':1,
    'Bent':2,
    'Compact':3,
}

dic_labels_rev = {
    0:'FR1',
    1:'FR2', 
    2:'Bent',
    3:'Compact',
}

hyper_params = {
    'epochs': 200,
    'lr': 1e-3,
    'bits': 72,
    'dataset': '../../data'
}

data_path = Path(hyper_params["dataset"])
output_dir = './output/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set parameters for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def binarize_data(file_path,dic_labels,split_train = True):
    if split_train:
        files = file_path
    else:
        files = get_image_files(file_path)
    train_dl = model.dls.test_dl(files)
    preds,y = model.get_preds(dl=train_dl)
    db_label = np.array([str(files[pth]).split('/')[4] for pth in range(len(files))])
    db_label = np.array([dic_labels[lbl] for x, lbl in enumerate(db_label)])
    
    return preds, db_label


def SimplifiedTopMap(rB, qB, retrievalL, queryL, topk):
    """
    Args:
        rB: binary codes of the training set - reference set,
        qB: binary codes of the query set,
        retrievalL: labels of the training set - reference set, 
        queryL: labels of the query set, and 
        topk: the number of top retrieved results to consider.

    rB = r_binary
    qB = q_binary
    retrievalL = train_label
    queryL = valid_label
    topk = 100
    """

    num_query = queryL.shape[0]
    mAP = [0] * num_query
 
    for i, query in enumerate(queryL):

        rel = (np.dot(query, retrievalL.transpose()) > 0)*1 # relevant train label refs.
        hamm = np.count_nonzero(qB[i] != rB, axis=1) #hamming distance
        ind = np.argsort(hamm) #  Hamming distances in ascending order.
        rel = rel[ind] #rel is reordered based on the sorted indices ind, so that it corresponds to the sorted Hamming distances.

        top_rel = rel[:topk] #contains the relevance values for the top-k retrieved results
        tot_relevant_sum = np.sum(top_rel) #total number of relevant images in the top-k retrieved results

        #skips the iteration if there are no relevant results.
        if tot_relevant_sum == 0:
            continue

        pr_num = np.linspace(1, tot_relevant_sum, tot_relevant_sum) 
        pr_denom = np.asarray(np.where(top_rel == 1)) + 1.0 #is the indices where top_rel is equal to 1 (i.e., the positions of relevant images)
        pr = pr_num / pr_denom # precision

        if (query == np.array([0, 0, 1, 0])).sum()==4:# Bent
            mAP_sub = np.sum(pr) /np.min(np.array([305,topk]))
        elif (query == np.array([0, 1, 0, 0])).sum()==4:# FRII
            mAP_sub = np.sum(pr) / np.min(np.array([434,topk]))
        elif (query == np.array([1, 0, 0, 0])).sum()==4:# FRI
            mAP_sub = np.sum(pr) /  np.min(np.array([215,topk]))
        else: # Compact
            mAP_sub = np.sum(pr) / np.min(np.array([226,topk]))

        mAP[i] = mAP_sub 
      
    return np.mean(mAP)*100, np.std(mAP)*100, mAP


def map_values(r_database, q_database, thresh=0.5, percentile=True, topk=100):
    """
    Calculate the mean Average Precision (mAP) for a given retrieval database.

    Args:
        r_database: Retrieval database object.
        q_database: Query database object.
        thresh: Threshold for binarizing predictions.
        percentile: Whether to use percentile for thresholding.
        topk: Number of top results to consider for mAP calculation.

    Returns:
        tuple: mAP, mAP standard deviation, mAP values, 
               binarized retrieval predictions, retrieval labels, 
               binarized query predictions, query labels.
    """

    if percentile:
        r_binary = np.array(
            [((out >= np.percentile(out, thresh)) * 1).tolist() 
             for _, out in enumerate(r_database.predictions)]
        )
        q_binary = np.array(
            [((out >= np.percentile(out, thresh)) * 1).tolist() 
             for _, out in enumerate(q_database.predictions)]
        )
    else:
        r_binary = np.array(
            [((out >= thresh) * 1).tolist() 
             for _, out in enumerate(r_database.predictions)]
        )
        q_binary = np.array(
            [((out >= thresh) * 1).tolist() 
             for _, out in enumerate(q_database.predictions)]
        )

    train_label = label_binarize(r_database.label_code, classes=[0, 1, 2,3])
    valid_label = label_binarize(q_database.label_code, classes=[0,1, 2,3])

    mAP, mAP_std, mAP_values1 = SimplifiedTopMap(
        r_binary, q_binary, train_label, valid_label, topk
    )

    return (
        mAP,
        mAP_std,
        mAP_values1,
        r_binary,
        train_label,
        q_binary,
        valid_label,
    )


# pre-process and load the data
data_block = DataBlock(
    blocks = (ImageBlock,CategoryBlock),
    get_items = get_image_files,
    splitter = GrandparentSplitter(),
    item_tfms=Resize(224),
    get_y = parent_label
)

data = data_block.dataloaders(data_path)

# define model
class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hd = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.2),
            nn.Linear(512, out_features),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.hd(x)
    

def DSHLoss(output, target, alpha = 0.00001, margin=36):
    # Move the input tensors to GPU
    output = output.to(device)
    target = target.to(device)
        
    #alpha = 1e-5  # m > 0 is a margin. The margin defines a radius around X1,X2. Dissimilar pairs contribute to the loss
    # function only if their distance is within this radius
    m = 2 * margin  # Initialize U and Y with the current batch's embeddings and labels
    target = target.int()
    

    # Create a duplicate y_label
    target = target.unsqueeze(1).expand(len(target), len(target))
    #y_label = torch.ones_like(torch.empty(len(target), len(target)))
    y_label = torch.ones_like(torch.empty(len(target), len(target))).to(device)
    y_label[target == target.t()] = 0

    #dist = torch.cdist(output, output, p=2).pow(2)
    dist = torch.cdist(output, output, p=2).pow(2).to(device)
    loss1 = 0.5 * (1 - y_label) * dist
    loss2 = 0.5 * y_label * torch.clamp(m - dist, min=0)

    #B1 = torch.norm(torch.abs(output) - 1, p=1, dim=1)
    B1 = torch.norm(torch.abs(output) - 1, p=1, dim=1).to(device)
    B2 = B1.unsqueeze(1).expand(len(target), len(target))

    loss3 = (B2 + B2.t()) * alpha
    minibatch_loss = torch.mean(loss1 + loss2 + loss3)
    return minibatch_loss


model = vision_learner(
    data,
    models.densenet161,
    custom_head = CustomHead(4416, hyper_params["bits"]),
    loss_func = DSHLoss,
    metrics = error_rate
)

# freeze the earlier layers and keep only the last layer trainable
model.freeze()

# train the custom head 
model.fit_one_cycle(1)

# unfreeze the model
model.unfreeze()

# train the model
model.fit_one_cycle(n_epoch = hyper_params["epochs"], lr_max=slice(hyper_params["lr"]),wd=1e-3)
model.recorder.plot_loss()
# plt.savefig(output_dir + "Densenet_Train_and_Test_Loss_curve_at_freeze.png")
# plt.close()

# # save the Learner object
# model.save('densenet_' + str(hyper_params['bits']) + 'bit_lr0p001')

# # save the model's state_dict instead of the entire Learner object
# torch.save(
#     model.model.state_dict(),
#     'densenet_' + str(hyper_params['bits']) + 'bit_lr0p001.pth'
# )

preds_train, train_label = binarize_data(
    file_path =  '../../data/train/',
    dic_labels = dic_labels,
    split_train = False
)

preds_test, test_label = binarize_data(
    file_path = '../../data/test/',
    dic_labels = dic_labels,
    split_train = False
)

df_testing = pd.DataFrame()
flat_predictions_test = []
for i in range(len(test_label)):
  flat_predictions_test.append(list(np.array(preds_test)[i]))

df_testing['predictions'] = flat_predictions_test
df_testing['label_code'] = test_label
df_testing['lable_name'] = df_testing['label_code'].map(dic_labels_rev)
df_testing.to_csv(output_dir +'/df_testing.csv',index = False)

df_training = pd.DataFrame()
flat_predictions_train = []
for i in range(len(train_label)):
  flat_predictions_train.append(list(np.array(preds_train)[i]))

df_training['predictions'] = flat_predictions_train
df_training['label_code'] = train_label
df_training['lable_name'] = df_training['label_code'].map(dic_labels_rev)
# df_training.to_csv(output_dir +'/df_training.csv',index = False)

thresholds_abs_values = np.arange(-1, 1.2, 0.1)
mAP_results_test = []
df_training1 = shuffle(df_training)

for _,thresh in enumerate(thresholds_abs_values):
    
    mAP_test_thresh, _, _, _, _, _, _ = map_values(df_training1, df_testing,thresh = thresh, percentile = False,topk=100)
    mAP_results_test.append(mAP_test_thresh)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(thresholds_abs_values, mAP_results_test, label='mAP_test')
plt.xlabel('Threshold')
plt.ylabel('mAP')
plt.legend()
plt.savefig(output_dir + '/Maps_curves_abs_values.png')
plt.close()

# Find the index of the maximum mAP value
data = {
    'mAP': mAP_results_test,
    'thresholds_abs_values': thresholds_abs_values
}

df = pd.DataFrame(data)

max_map_index = df['mAP'].idxmax()

# Retrieve the threshold corresponding to the maximum mAP
threshold_max_map = df.loc[max_map_index, 'thresholds_abs_values']

maP_test, _, _, _, _, _, _ = map_values(df_training1, df_testing, thresh = threshold_max_map, percentile = False, topk=100)

# Create a dictionary with the variable names and their values
data = {
    'optimal threshold': [threshold_max_map],
    'Test mAP': [maP_test]
}

# Create the DataFrame
df_results = pd.DataFrame(data)
df_results.to_csv(output_dir + '/final_results.csv', index = False)

