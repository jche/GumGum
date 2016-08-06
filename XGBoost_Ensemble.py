import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import os
import csv
import time


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(data):
    print "Reading Data..."
    temp = np.load(data)
    d = csr_matrix((temp['data'], temp['indices'], temp['indptr']), shape=temp['shape'], dtype=float).toarray()
    return d


def format_data(data):
    d = get_data(data)
    m = int(np.size(d,1))   # Number of columns
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y


def net_sav(r,f):
    sav = -5200+127000*f-850000*(1-r)
    return sav


def process_data(month, day, hour):
    root = "/mnt/rips2/2016/"
    p1 = str(month).rjust(2, "0")
    p2 = str(day).rjust(2, "0")
    p3 = str(hour).rjust(2, "0")

    # Inputting training and testing set
    name = os.path.join(root, p1, p2, p3, "output_new.npy")
    print "Processing " + name
    data, label = format_data(name)
    matrix = xgb.DMatrix(data, label=label)
    return matrix, label


def train_model(month, day, hour):
    dtrain, train_label = process_data(month, day, hour)

    p = sum(train_label)   # number of ones
    n = len(train_label) - p   # number of zeros
    # Setting parameters
    param = {'booster': 'gbtree',   # Tree, not linear regression
             'objective': 'binary:logistic',   # Output probabilities
             'eval_metric': ['auc'],
             'bst:max_depth': 5,   # Max depth of tree
             'bst:eta': .2,   # Learning rate (usually 0.01-0.2)
             'bst:gamma': 8.5,   # Larger value --> more conservative
             'bst:min_child_weight': 1,
             'scale_pos_weight': n/float(p),   # Often num_neg/num_pos
             'subsample': .8,
             'silent': 1,   # 0 outputs messages, 1 does not
             'save_period': 0,   # Only saves last model
             'nthread': 6,   # Number of cores used; otherwise, auto-detect
             'seed': 30}
    evallist = [(dtrain,'train')]   # Want to train until eval error stops decreasing
    num_round = 250   # Number of rounds of training
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist)
    return bst



def eval_quality(single_pred, ens_pred, ens_pred_sum, test_label):
    quality = 0
    vals = [float(len(list))]*5
    p1 = np.sum(ens_pred, axis=0)/vals

    pc = 1#percentage correct, which is p1 if ensemble is correct
    pt = 1# percentage that correspond to new tree

    for i in range(0, length):
        if single_pred[i] != test_label[i]:
            quality = quality - (1-abs(pc[i]-pt[i]))
        elif ens_pred_sum[i] != test_label[i]:
            quality = quality + (1-abs(p1[i]-pc[i]))
        else:
            quality = quality + (1+abs(p1[i]-p2[i]))
    return quality


# def write_to_file(predictions, elapsed, addr):
#     with open(addr) as file:
#         # J score, AUC score, best recall, best filter rate, best cutoff
#         results = [0, 0, 0, 0, elapsed]
#         for cutoff in range(0, 31):
#             cut = cutoff/float(100)   # Cutoff in decimal form
#             y = predictions > cut   # If y values are greater than the cutoff
#             recall = metrics.recall_score(test_label, y)
#             filter_rate = sum(np.logical_not(y))/float(len(y_pred))
#             if net_sav(recall, filter_rate) > results[0]:
#                 results[0] = net_sav(recall, filter_rate)
#                 results[1] = recall
#                 results[2] = filter_rate
#                 results[3] = cut
#         wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
#         wr.writerow(results)


#bst.save_model('/home/rmendoza/Desktop/xgbtemp.model')


if __name__ == "__main__":
    ensemble_cap = 7
    ensemble = []
    ensemble_qual = []
    for month in range(6, 7):
        for day in range(4, 5):
            for hour in range(0, 24):
                bst = train_model(month, day, hour)   # C_i-1
                if hour == 23:
                    dtest, test_label = process_data(month, day+1, 0)
                else:
                    dtest, test_label = process_data(month, day, hour+1)

                single_pred = bst.predict(dtest)
                ens_pred = []
                for model in ensemble:
                    ens_pred.append(model.predict(dtest))

                # Cutoff function
                cut = .15
                length = len(single_pred)
                single_pred = single_pred > cut
                for pred in ens_pred:
                    pred = pred > cut
                ens_pred_prop = np.sum(ens_pred, axis=0)/([float(len(ens_pred))]*length)
                ens_pred_round = np.round(ens_pred_prop)   # Rounds down if exactly .5

                # Check out how the predictions for the next day are (Conf. Matrix)
                recall = metrics.recall_score(test_label, ens_pred_round)
                filter_rate = sum(np.logical_not(ens_pred_round))/float(length)
                print "Recall is %s" % recall
                print "Filter rate is %s" % filter_rate
                print "Savings are %s" % net_sav(recall, filter_rate)

                # Evaluate quality
                quality = 0
                p1 = np.abs(np.logical_not(ens_pred_round)-ens_pred_prop)   # Prop in majority vote
                p2 = np.ones(length) - p1   # Prop in minority vote
                pc = np.abs(ens_pred_prop + (test_label-np.ones(length)))   # Prop correct
                pt = np.abs(ens_pred_prop + (single_pred-np.ones(length)))   # Prop same as single_pred
                for i in range(0, length):
                    if single_pred[i] != test_label[i]:
                        quality = quality - (1-abs(pc[i]-pt[i]))
                    elif ens_pred_round[i] != test_label[i]:
                        quality = quality + (1-abs(p1[i]-pc[i]))
                    else:
                        quality = quality + (1+abs(p1[i]-p2[i]))

                # Add to ensemble
                if len(ensemble) < ensemble_cap:
                    ensemble.append(bst)
                    ensemble_qual.append(quality)
                else:
                    if quality > min(ensemble_qual):
                        index = ensemble_qual.index(min(ensemble_qual))
                        ensemble[index] = bst
                        ensemble_qual[index] = quality
