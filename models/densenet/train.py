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
predictions_dir = './predictions/'
binaries_dir = './binaries/'
figures_dir = './figures/'

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

# pre-process and load the data
data_block = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = GrandparentSplitter(),
    item_tfms=Resize(224),
    get_y = parent_label
)


def generate_predictions(model, file_path, dic_labels):

    files = get_image_files(file_path)
    train_dl = model.dls.test_dl(files)
    preds, _ = model.get_preds(dl=train_dl)
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


def run(bitsize=72, learning_rate=1e-3, model_name='densenet_72bit_lr0p001', save_model=False):

    data = data_block.dataloaders(data_path)

    model = vision_learner(
        data,
        models.densenet161,
        custom_head = CustomHead(4416, bitsize),
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
    model.fit_one_cycle(
        n_epoch = hyper_params["epochs"], 
        lr_max = slice(learning_rate), 
        wd = learning_rate)
    
    model.recorder.plot_loss()
    # plt.savefig(output_dir + "Densenet_Train_and_Test_Loss_curve_at_freeze.png")
    # plt.close()

    # Experiment 5 code begins
    
    import time

    def measure_hashing_time(model, dataloader):
        """
        Measures the average hashing time for a given model and dataloader.

        Args:
            model: The trained deep hashing model.
            dataloader: The dataloader for the test dataset.

        Returns:
            float: The average hashing time in seconds.
        """

        total_time = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images, _ = batch
                images = images.to(device)

                start_time = time.time()
                _ = model.model(images)  # Generate hash codes
                end_time = time.time()

                total_time += (end_time - start_time)
                total_samples += len(images)

        avg_hashing_time = total_time / total_samples
        return avg_hashing_time

    # Create a dataloader for the test dataset
    test_files = get_image_files('../../data/test/')
    test_dl = model.dls.test_dl(test_files)

    # Measure the hashing time
    avg_hashing_time = measure_hashing_time(model, test_dl)

    # Print the average hashing time
    print(f"Average Hashing Time: {avg_hashing_time:.4f} seconds")

    # Write the hashing time to a text file
    with open("hashing_time.txt", "w") as f:
        f.write(f"Average Hashing Time: {avg_hashing_time:.4f} seconds")

    # Experiment code ends

    if save_model:

        # save the Learner object
        model.save('densenet_' + str(bitsize) + 'bit_lr0p001')

        # save the model's state_dict
        torch.save(
            model.model.state_dict(),
            'densenet_' + str(bitsize) + 'bit_lr0p001.pth'
        )
    
    # Generate and save the predictions
    train_predictions, train_labels = generate_predictions(
        model=model,
        file_path='../../data/train/',
        dic_labels=dic_labels,
    )

    test_precitions, test_labels = generate_predictions(
        model=model,
        file_path='../../data/test/',
        dic_labels=dic_labels,
    )

    testing_df = pd.DataFrame()
    flat_predictions_test = []
    for i in range(len(test_labels)):
        flat_predictions_test.append(list(np.array(test_precitions)[i]))

    testing_df['predictions'] = flat_predictions_test
    testing_df['label_code'] = test_labels
    testing_df['lable_name'] = testing_df['label_code'].map(dic_labels_rev)
    testing_df.to_csv(predictions_dir + model_name + '.csv', index = False)

    training_df = pd.DataFrame()
    flat_predictions_train = []
    for i in range(len(train_labels)):
        flat_predictions_train.append(list(np.array(train_predictions)[i]))

    training_df['predictions'] = flat_predictions_train
    training_df['label_code'] = train_labels
    training_df['lable_name'] = training_df['label_code'].map(dic_labels_rev)
    # df_training.to_csv(output_dir +'/df_training.csv',index = False)

    thresholds_abs_values = np.arange(-1, 1.2, 0.1)
    mAP_results_test = []
    training_df_shuffled = shuffle(training_df)

    for _,thresh in enumerate(thresholds_abs_values):
        
        mAP_test_thresh, _, _, _, _, _, _ = map_values(
            training_df_shuffled, 
            testing_df,
            thresh=thresh, 
            percentile=False,
            topk=100
        )
        mAP_results_test.append(mAP_test_thresh)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_abs_values, mAP_results_test, label='mAP_test')
    plt.xlabel('Threshold')
    plt.ylabel('mAP')
    plt.legend()
    plt.savefig(figures_dir + model_name + '_maps_curves.png')
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

    test_mAP, _, _, _, _, _, _ = map_values(
        training_df_shuffled,
        testing_df, 
        thresh=threshold_max_map, 
        percentile=False, 
        topk=100
    )

    # Create a dictionary with the variable names and their values
    data = {
        'optimal threshold': [threshold_max_map],
        'Test mAP': [test_mAP]
    }

    # Create the DataFrame
    df_results = pd.DataFrame(data)
    df_results.to_csv(output_dir + model_name + '_mAP.csv', index = False)


if __name__ == '__main__':

    bitsizes = [56, 72]

    for bitsize in bitsizes:
        run(bitsize=bitsize, 
            learning_rate=1e-3, 
            model_name='densenet_' + str(bitsize) + 'bit_lr0p001')
