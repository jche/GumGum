import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, recall_score
import operator
import pandas as pd
from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder


# XGBoost 101 found at http://xgboost.readthedocs.io/en/latest/python/python_intro.html


def get_data(data):
    print "Reading Data..."
    temp = np.load(data)
    d = csr_matrix((temp['data'], temp['indices'], temp['indptr']), shape=temp['shape'], dtype=float).toarray()
    return d


def format_data(data):
    d = get_data(data)
    m = int(np.size(d,1))   # Number of columns
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m-1)
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y


def recall(preds, dtrain):
    cutoff = 0.1
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    preds_bin = np.greater(preds, np.zeros(len(labels))+cutoff)
    return "recall", recall_score(labels, preds_bin)


if __name__ == "__main__":
    # Inputting training and testing set
    train_data, train_label = format_data("/home/jche/Data/day_samp_bin_0604.npy")
    dtrain = xgb.DMatrix(train_data, label=train_label)
    test_data, test_label = format_data("/home/jche/Data/day_samp_bin_0605.npy")
    dtest = xgb.DMatrix(test_data, label=test_label)

    # Setting parameters
    param = {'booster':'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'bst:max_depth':4,   # Max depth of tree
             'bst:eta':.5,   # Learning rate (usually 0.01-0.2)
             'silent':0,   # 0 outputs messages, 1 does not
             'nthread':4}    # Number of cores used; otherwise, auto-detect
    #param['eval_metric'] = 'error'
    evallist = [(dtest,'eval'), (dtrain,'train')]

    num_round = 100   # Number of rounds of training, increasing this increases the range of output values
    #bst = xgb.train(param, dtrain, num_round, evallist, feval=recall, maximize=True)
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
    bst.dump_model('dump.raw.txt')

    y_true = test_label
    y_pred = bst.predict(dtest)
    for cutoff in range(1, 10):
        cut = cutoff/float(10)   # Cutoff, checking from .1 thru .9
        print "Cutoff is: %s" % cut
        y = np.greater(y_pred, np.zeros(len(y_true))+cut)   # If y values are greater than the cutoff
        print "Recall is: %s" % recall_score(y_true, y)
        print confusion_matrix(y_true, y)

    #xgb.plot_importance(bst, xlabel="test")
    #xgb.plot_tree(bst, num_trees=2)