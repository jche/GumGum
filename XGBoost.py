import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import time
try:
    import cPickle as pickle
except:
    import pickle


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
    x = d[:200000, :m-1]   # To reduce load on computer
    y = d[:200000, m-1]
    return x, y


if __name__ == "__main__":
    # Inputting training and testing set
    train_data, train_label = format_data("/home/jche/Data/alldata3_bin.npy")
    dtrain = xgb.DMatrix(train_data, label=train_label)
    test_data, test_label = format_data("/home/jche/Data/alldata5_bin.npy")
    dtest = xgb.DMatrix(test_data, label=test_label)

    # Setting parameters
    param = {'booster':'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'bst:max_depth':5,   # Max depth of tree
             'bst:eta':.1,   # Learning rate (usually 0.01-0.2)
             'bst:gamma': 1,   # Larger value --> more conservative
             'silent':1,   # 0 outputs messages, 1 does not
             'save_period':0,   # Only saves last model
             'nthread':8}    # Number of cores used; otherwise, auto-detect
    param['eval_metric'] = ['error', 'logloss']
    evallist = [(dtest,'eval'), (dtrain,'train')]

    num_round = 100   # Number of rounds of training, increasing this increases the range of output values
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
    bst.dump_model('/home/jche/Desktop/dump.raw.txt')
    # bst.save_model('/home/jche/Desktop/0001.model')
    # bst = xgb.Booster({'nthread':4}) #init model
    # bst.load_model("model.bin") # load data

    y_true = test_label
    y_pred = bst.predict(dtest)
    for cutoff in range(0, 31):
        cut = cutoff/float(100)   # Cutoff in decimal form
        print "Cutoff is: %s" % cut
        y = y_pred > cut   # If y values are greater than the cutoff
        print "Recall is: %s" % metrics.recall_score(y_true, y)
        print metrics.confusion_matrix(y_true, y)
