import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import os
import csv


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


if __name__ == "__main__":
    root = "/mnt/rips2/2016/"
    # Setting parameters
    param = {'booster': 'gbtree',   # Tree, not linear regression
             'objective': 'binary:logistic',   # Output probabilities
             'eval_metric': ['auc'],
             'bst:max_depth': 5,   # Max depth of tree
             'bst:eta': .2,   # Learning rate (usually 0.01-0.2)
             'bst:gamma': 0,   # Larger value --> more conservative
             'bst:min_child_weight': 1,
             'scale_pos_weight': 30,   # Often num_neg/num_pos
             'subsample': .8,
             'silent': 1,   # 0 outputs messages, 1 does not
             'save_period': 0,   # Only saves last model
             'nthread': 6,   # Number of cores used; otherwise, auto-detect
             'seed': 25}
    for month in range(5, 7):
        p1 = str(month).rjust(2, "0")
        for day in range(4, 26):
            p2 = str(day).rjust(2, "0")
            p3 = str(day+1).rjust(2, "0")
            try:
                # Inputting training and testing set
                train_data_name = os.path.join(root, p1, p2, "day_samp_new.npy")
                train_data, train_label = format_data(train_data_name)
                dtrain = xgb.DMatrix(train_data, label=train_label)

                test_data_name = os.path.join(root, p1, p3, "day_samp_new.npy")
                test_data, test_label = format_data(test_data_name)
                dtest = xgb.DMatrix(test_data, label=test_label)

                print "Working on " + train_data_name

                evallist = [(dtrain,'train'), (dtest,'eval')]   # Want to train until eval error stops decreasing

                num_round = 1000   # Number of rounds of training, increasing this increases the range of output values
                bst = xgb.train(param,
                                dtrain,
                                num_round,
                                evallist,
                                early_stopping_rounds=10)   # If error doesn't decrease in n rounds, stop early

                root2 = "/home/ubuntu/Jonathan"
                model_name = os.path.join(root2, "xgb_model" + p1 + p2)
                bst.dump_model(model_name)

                y_pred = bst.predict(dtest)

                with open(os.path.join(root2, "xgb_numbers.csv"), "a") as file:
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
                    wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
                    wr.writerow(results)
            except:
                pass
