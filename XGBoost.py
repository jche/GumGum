import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics


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


if __name__ == "__main__":
    # Inputting training and testing set
    train_data, train_label = format_data("/home/jche/Desktop/day_samp_res_50_0604.npy")
    dtrain = xgb.DMatrix(train_data, label=train_label)
    test_data, test_label = format_data("/home/jche/Desktop/day_samp_res_50_0605.npy")
    dtest = xgb.DMatrix(test_data, label=test_label)

    # Setting parameters
    param = {'booster':'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'eval_metric':['auc'],
             'bst:max_depth':5,   # Max depth of tree
             'bst:eta':.1,   # Learning rate (usually 0.01-0.2)
             'bst:gamma':0,   # Larger value --> more conservative
             'bst:min_child_weight':1,
             'scale_pos_weight':30,   # Often num_neg/num_pos
             'subsample':.8,
             'silent':1,   # 0 outputs messages, 1 does not
             'save_period':0,   # Only saves last model
             'nthread':6,   # Number of cores used; otherwise, auto-detect
             'seed':25}
    evallist = [(dtrain,'train'), (dtest,'eval')]

    num_round = 1000   # Number of rounds of training, increasing this increases the range of output values
    bst = xgb.train(param,
                    dtrain,
                    num_round,
                    evallist,
                    early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early
    bst.dump_model('/home/jche/Desktop/xgb_june_04_to_05_v2.txt')

    y_true = test_label
    y_pred = bst.predict(dtest)
    # J score, AUC score, best recall, best filter rate, best cutoff
    results = [0, 0, 0, 0, 0]
    for cutoff in range(0, 31):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y = y_pred > cut   # If y values are greater than the cutoff
        recall = metrics.recall_score(test_label, y)
        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
        filter_rate = sum(np.logical_not(y))/float(len(y_pred))
        if recall*6.7+filter_rate > results[0]:
            results[0] = recall*6.7+filter_rate
            results[1] = metrics.roc_auc_score(test_label, y)
            results[2] = recall
            results[3] = filter_rate
            results[4] = cut
    print results
