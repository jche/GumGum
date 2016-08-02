import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.naive_bayes import BernoulliNB
import time
import os
import csv
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
    print 'm = ', m
        #Get rid of column 2460,, of d that is not bernoulli
    x = np.delete(x,np.s_[2460],1)
    # for counter in range(m-2):
    #     if x[1,counter] != 0:
    #         if x[1,counter]!=1:
    #             print 'A non-bernoulli variable is found! ... '
    #             print 'counter = ', counter
    #             print 'x[1, counter] = ', x[1, counter]
    #             #x = np.delete(x,np.s_[counter],1)
    # print 'wait 20 seconds'
    # time.sleep(20)

    # Just checking that indeed all non-Bernoulli variables are deleted
    for counter in range(m-2):
        if x[1,counter] != 0:
            if x[1,counter]!=1:
                print 'Oh no! Still a nonbernoulli variable is found! ... '
                print 'counter = ', counter
                print 'x[1, counter] = ', x[1, counter]

    return x, y


def recall(preds, dtrain):
    cutoff = 0.1
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    preds_bin = np.greater(preds, np.zeros(len(labels))+cutoff)
    return "recall", recall_score(labels, preds_bin)

def GetData(month, day): ## Input Weiyi-formatted Data
    """
    Takes the data from a given day/month and outputs a numpy array
    :param month:
    :param day:
    :return:
    """
    #root = "/mnt/rips2/2016"  #for AWS
    # root = "/home/rmendoza/Documents/Data/DataXGB_jul28"  #for local maschine
    root = "/home/rmendoza/Documents/Data/DataReservoir_aug1"
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    #dataroot = os.path.join(root,p0,p1,"day_samp_bin.npy")  # for AWS
    #binName = 'day_samp_bin'+p0+p1+'.npy'  #for local maschine #old data
    #binName = 'day_samp_new_'+p0+p1+'.npy'## New data
    binName = 'day_samp_res_25__'+p0+p1+'.npy'## Last data data day_samp_res_25__0604
    dataroot = os.path.join(root,binName)   #for local maschine
    print "Reading Data..."
    train_data, train_label = format_data(dataroot)

    #temp = np.load(dataroot)  #old code
    #Data = csr_matrix((  temp['data'], temp['indices'], temp['indptr']),shape = temp['shape'], dtype=float).toarray()

    print "Finished reading data file"
    return train_data, train_label

def netSav(r,f):
    netSaving = -5200+127000*f-850000*(1-r)
    return netSaving

alfa = 0.00005
clf = BernoulliNB(class_prior=[alfa, 1-alfa])
if __name__ == "__main__":
    #with open("/home/rmendoza/Desktop/resultsNB_reservoir_50.ods", "w") as output_file:
    with open("/home/rmendoza/Documents/NaiveBayesianDocumentation/Aug1_resultsToDate/resultsNB_reservoir_25_alfa0?00005.ods", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        l = ['alfa = ',alfa]
        wr.writerow(l)
        l = ['month','TrainDay','testDay','recall','filtered', 'netSaving', 'time']
        wr.writerow(l)
        for diff in [1]:  #1,7  # as for now, only [1] means test on next day
            for month in range(6,7): #5,7    # as for now, only range(6,7) means june
                for day in range(4,26): #1,32  # as for now, only range(4,5) means 1st day
                    print '------------------------------------------------'
                    print '------------------------------------------------'
                    print 'month = ', month,' and day = ',  day
                    try:
                        start_time = time.time()
                        # Inputting training and testing set
                        train_data, train_label = GetData(month, day)
                        test_data, test_label = GetData(month, day+diff)
                        print 'Data Read'
                        #time.sleep(20)  #sleep
                        print 'Training Data...'
                        clf.partial_fit(train_data, train_label, classes=[0, 1])
                        print 'Data Trained...'
                        y_true = test_label
                        n = len(y_true)
                        ### Here's a problem...
                        #print 'predictin jo ...'
                        y_pred = clf.predict(test_data)
                        #print 'getting conf matrix...'
                        cf = confusion_matrix(y_true,y_pred)
                        #print 'calculating recall...'
                        recalll = recall_score(y_true, y_pred)
                        #print 'calculating filtering'
                        filtered = (cf[0,0])/float(n)
                        netSaving=12700*filtered-5200-850000*(1-recalll)
                        print "Recall is: %s" % recalll
                        print 'Filtering is = ', filtered
                        print 'netSaving is = ', netSaving
                        print cf
                        #get time:
                        timer = time.time() - start_time
                        print 'Time = ', timer
                        testday = day + diff
                        l = [month,day,testday,recalll,filtered, netSaving,timer]
                        wr.writerow(l)

                    except:
                        pass
                        print 'failure, no such day (or some other potential error)'
                        #time.sleep(20)  #sleep
                    print '_____________________________________________________________________'
                    print '_____________________________________________________________________'