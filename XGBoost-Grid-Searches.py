import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
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
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m-1)
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y


if __name__ == "__main__":
    with open('/home/kbhalla/Desktop/Results/XGB-Grid-Results3.csv', 'w') as file:
        # Inputting training and testing set
        wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
        wr.writerow(['J-score','AUC','Recall','Filter','Cut','Net Savings', 'eta', 'alpha', 'day trained', 'day predicted'])
        for eta in [.01, .05, .1, .15, .2]:
            for alpha in [0,1,2,4]:
                for i in range(22,25):
                    p0 = str(i).rjust(2,'0')
                    p1 = str(i+1).rjust(2,'0')
                    #train_data, train_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p0+".npy")
                    train_data, train_label = format_data('/home/rmendoza/Documents/Data/DataXGB_jul28/day_samp_new_06'+p0+'.npy')
                    dtrain = xgb.DMatrix(train_data, label=train_label)
                    #test_data, test_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p1+".npy")
                    test_data, test_label = format_data('/home/rmendoza/Documents/Data/DataXGB_jul28/day_samp_new_06'+p1+'.npy')
                    dtest = xgb.DMatrix(test_data, label=test_label)
                    p = np.count_nonzero(train_label)
                    n = len(train_label) - p
                    # Setting parameters
                    param = {'booster':'gbtree',   # Tree, not linear regression
                             'objective':'binary:logistic',   # Output probabilities
                             'eval_metric':['auc'],
                             'bst:max_depth':5,   # Max depth of tree
                             'bst:eta':eta,   # Learning rate (usually 0.01-0.2)
                             'bst:gamma':8.5,   # Larger value --> more conservative
                             'bst:min_child_weight':1,
                             'scale_pos_weight':n/float(p),   # Often num_neg/num_pos
                             'subsample':.9,
                             'silent':1,   # 0 outputs messages, 1 does not
                             'save_period':0,   # Only saves last model
                             'nthread':6,   # Number of cores used; otherwise, auto-detect
                             'seed':25,
                             'alpha': alpha}
                    evallist = [(dtrain,'train'), (dtest,'eval')]

                    num_round = int(100*0.2/float(eta))   # Number of rounds of training, increasing this increases the range of output values
                    bst = xgb.train(param,
                                    dtrain,
                                    num_round,
                                    evallist)   # If error doesn't decrease in n rounds, stop early
                    bst.dump_model('/home/kbhalla/Desktop/xgb/xgb_june_' + p0 + '_to_' + p1 +'_v2.txt')

                    y_true = test_label
                    y_pred = bst.predict(dtest)
                    # J score, AUC score, best recall, best filter rate, best cutoff
                    results = [0, 0, 0, 0, 0, 0, eta, alpha, p0, p1]
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
                            results[5] = 127000*filter_rate -5200 -850000*(1-recall)
                    wr.writerow(results)
                wr.writerow(['Increasing Child'])
            wr.writerow(['Increasing Subsample','Child'])
