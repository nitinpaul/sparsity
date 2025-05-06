
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from fastai.data.all import *
from fastai.vision.all import *
from fastai.data.transforms import get_image_files
from matplotlib import pyplot as plt

from .utils import get_verified_data


dic_labels = { 
    'FRI':0,
    'FRII':1,
    'Bent':2,
    'Compact':3,
}

dic_labels_rev = {
    0:'FRI',
    1: 'FRII', 
    2:'Bent',
    3:'Compact',
}


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


def map_at_k_equals_r(tr_df, tst_or_vl_df, thresh):
    """
    Calculates the mean Average Precision (mAP) at k equals R.

    Args:
        tr_df: DataFrame containing training data.
        tst_or_vl_df: DataFrame containing test or validation data.
        thresh: Threshold value.

    Returns:
        float: The mean average precision value rounded to 2 decimal places.
    """

    # Bent
    topk = 305
    map_bent, _, _, _, _, _, _ = map_values(
        tr_df, tst_or_vl_df, thresh, percentile=False, topk=topk
    )

    # FRII
    topk = 434
    map_frii, _, _, _, _, _, _ = map_values(
        tr_df, tst_or_vl_df, thresh, percentile=False, topk=topk
    )

    # FRI
    topk = 215
    map_fri, _, _, _, _, _, _ = map_values(
        tr_df, tst_or_vl_df, thresh, percentile=False, topk=topk
    )

    # Compact
    topk = 226
    map_comp, _, _, _, _, _, _ = map_values(
        tr_df, tst_or_vl_df, thresh, percentile=False, topk=topk
    )
    
    mean_average = round(
        np.mean(np.array([map_bent, map_frii, map_fri, map_comp])), 2
    )
    
    return mean_average


def evaluate_cosfire_accuracy(model, data):
    """
    Runs the accuracy experiment on the given cosfire model and data.

    Args:
        model: The model to evaluate.
        data: The data module containing the datasets.

    Returns:
        tuple: A tuple containing the maximum mAP value, the threshold 
               at which the maximum mAP is achieved, and the predictions.
    """

    # Set model to evaluation mode
    model.eval()

    # Set params
    batch_size = 32

    # Set path to descriptors
    full_data_path = './models/cosfire/descriptors/train_valid_test.mat'
    test_data_path = './models/cosfire/descriptors/train_test.mat' 
    validation_data_path = './models/cosfire/descriptors/train_valid.mat'

    train_df, _, test_df = get_verified_data(
        full_data_path, validation_data_path, test_data_path, dic_labels
    )

    train_dataset = data.CosfireDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = data.CosfireDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Lists to store predictions
    predictions_train = []
    predictions_test = []

    # Make predictions
    with torch.no_grad():

        for _, (train_inputs, _) in enumerate(train_dataloader):
            train_outputs, _ = model(train_inputs)
            predictions_train.append(train_outputs.numpy())
        
        for _, (test_inputs, _) in enumerate(test_dataloader):
            test_outputs, _ = model(test_inputs)
            predictions_test.append(test_outputs.numpy())

    # Flatten the predictions
    flat_predictions_train = [item for sublist in predictions_train for item in sublist]
    flat_predictions_test = [item for sublist in predictions_test for item in sublist]
    
    # Append predictions to the DataFrames
    train_df['predictions'] = flat_predictions_train
    train_df['label_name'] = train_df['label_code'].map(dic_labels_rev)

    test_df['predictions'] = flat_predictions_test
    test_df['label_name'] = test_df['label_code'].map(dic_labels_rev)

    # Save predictions for binarization
    predictions = test_df[['predictions', 'label_name']]

    thresholds = np.arange(-1, 1.2, 0.1)
    mAP_values = []

    for _, threshold in enumerate(thresholds):
        mAP_at_threshold = map_at_k_equals_r(
            tr_df=train_df,
            tst_or_vl_df=test_df,
            thresh=threshold
        )                                       
        mAP_values.append(mAP_at_threshold)

    # Find the index of the maximum mAP value
    mAP_and_threshold_df = pd.DataFrame(
        {
            'mAP_value': mAP_values,
            'threshold': thresholds
        }
    )
    
    max_mAP_index = mAP_and_threshold_df['mAP_value'].idxmax()
    threshold_max_mAP = mAP_and_threshold_df.loc[max_mAP_index, 'threshold']

    return max(mAP_values), threshold_max_mAP, predictions

