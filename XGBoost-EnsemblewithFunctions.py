import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import csv
from SendEmail import sendEmail
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
    n = int(np.size(d,0))   # Number of rows
    print "There are %s data points, each with %s features" % (n, m-1)
    x = d[:, :m-1]
    y = d[:, m-1]
    return x, y

def getBst(dtrain, evallist, train_label, modelName,num_round, eta,dumpname):
    p = np.count_nonzero(train_label)
    n = len(train_label) - p
    # Setting parameters
    # Train Model 1...
    try:
        bst = xgb.Booster() #init model
        print 'Loading the model... '
        #print '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2.model'
        bst.load_model(modelName) # load model
        return bst
    except Exception as e:
        print e
        #time.sleep(30)
        print "Some mistake in loading the model, so we should better train..." #pass  #to skip
        param = {'booster':'gbtree',   # Tree, not linear regression
             'objective':'binary:logistic',   # Output probabilities
             'eval_metric':['auc'],
             'bst:max_depth':5,   # Max depth of tree
             'bst:eta':eta,   # Learning rate (usually 0.01-0.2)
             'bst:gamma':8.5,   # Larger value --> more conservative
             'bst:min_child_weight':1,
             'scale_pos_weight':n/float(p),   # Often num_neg/num_pos
             'subsample':.8,
             'silent':1,   # 0 outputs messages, 1 does not
             'save_period':0,   # Only saves last model
             'nthread':6,   # Number of cores used; otherwise, auto-detect
             'seed':25}
        #num_round = int(100*0.2/float(eta))   # Number of rounds of training, increasing this increases the range of output values
        bst = xgb.train(param,
                    dtrain,
                    num_round)#,
                    #evallist)   # If error doesn't decrease in n rounds, stop early#
        bst.save_model(modelName)
        print 'model saved'
        bst.dump_model(dumpname)
        print 'model dumped'
        return bst


def getAyAByB(y_hatTrain,train_data,train_label):
    ## Function recieves the y predicted for the train matrix and gives the x and y for the 0s and 1s
    ####y_hatTrain = bst.predict(dtrain)  #contains the 0s and 1s predicted by Model 1
    y_predTrain = []
    savings = [0]
    dacut = 0.09
    y = y_hatTrain > dacut
    recall = metrics.recall_score(train_label, y)
    filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
    if 127000*filter_rate -5200 -850000*(1-recall) > 0:
        savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
    #print 'beginn cutoffs'
    for cutoff in range(0, 20):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y = y_hatTrain > cut   # If y values are greater than the cutoff
        # print y
        recall = metrics.recall_score(train_label, y)
        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
        filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
        if 127000*filter_rate -5200 -850000*(1-recall) > savings[0]:
            savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
            dacut = cut
    #print 'serious binarize'
    for kk in range(len(y_hatTrain)):
        if y_hatTrain[kk] > dacut:
            y_predTrain.append(1)
        else:
            y_predTrain.append(0)
    pred_pos = []
    pred_neg = []
    #print 'done binirize'
    for i in range(len(y_hatTrain)):
        entry = train_data[i].tolist()
        entry.append(train_label[i])
        if y_predTrain[i] == 1:
            entry.append(1)
            pred_pos.append(entry)
        else:
            entry.append(0)
            pred_neg.append(entry)
    pred_pos = np.array(pred_pos)
    pred_neg = np.array(pred_neg)
    #print pred_pos
    #print 'pred_pos[0:20,-10:] = '
    #print pred_pos[0:20,-10:]
    A = pred_neg[:,:-2]   #new A to train on
    yA = pred_neg[:,-2]   # new testA
    B = pred_pos[:,:-2]   # new B to train on
    yB = pred_pos[:,-2]   # new testB to train on ...
    return A, yA, B, yB

def getAyAByBtest(y_hatTrain,train_data,train_label):
    ## Function recieves the y predicted for the test matrix and gives the x and y for the 0s and 1s
    ####y_hatTrain = bst.predict(dtrain)  #contains the 0s and 1s predicted by Model 1
    y_predTrain = []
    savings = [0]
    dacut = 0.09
    y = y_hatTrain > dacut
    recall = metrics.recall_score(train_label, y)
    filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
    if 127000*filter_rate -5200 -850000*(1-recall) > 0:
        savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
    #print 'beginn cutoffs'
    for cutoff in range(0, 20):
        cut = cutoff/float(100)   # Cutoff in decimal form
        y = y_hatTrain > cut   # If y values are greater than the cutoff
        # print y
        recall = metrics.recall_score(train_label, y)
        # true_negative_rate = sum(np.logical_not(np.logical_or(test_label, y)))/float(len(y_pred))
        filter_rate = sum(np.logical_not(y))/float(len(y_hatTrain))
        if 127000*filter_rate -5200 -850000*(1-recall) > savings[0]:
            savings[0] = 127000*filter_rate -5200 -850000*(1-recall)
            dacut = cut
    #print 'serious binarize'
    for kk in range(len(y_hatTrain)):
        if y_hatTrain[kk] > dacut:
            y_predTrain.append(1)
        else:
            y_predTrain.append(0)
    pred_pos = []
    pred_neg = []
    #print 'done binirize'
    for i in range(len(y_hatTrain)):
        entry = train_data[i].tolist()
        entry.append(train_label[i])
        if y_predTrain[i] == 1:
            entry.append(1)
            pred_pos.append(entry)
        else:
            entry.append(0)
            pred_neg.append(entry)
    pred_pos = np.array(pred_pos)
    pred_neg = np.array(pred_neg)
    #print pred_pos
    #print 'pred_pos[0:20,-10:] = '
    #print pred_pos[0:20,-10:]
    A = pred_neg[:,:-2]   #new A to train on
    yA = pred_neg[:,-2]   # new testA
    B = pred_pos[:,:-2]   # new B to train on
    yB = pred_pos[:,-2]   # new testB to train on ...
    return A, yA, B, yB

errorCounter = 0
if __name__ == "__main__":
    with open('/home/rmendoza/Desktop/XGBoost/XGB-Ensemble_1.csv', 'w') as file:
        try:
            # Inputting training and testing set
            wr = csv.writer(file, quoting = csv.QUOTE_MINIMAL)
            wr.writerow(['Net_Savings', 'num_round', 'day_trained', 'day_predicted','hour_trainedAndTested'])
            for i in range(4,25):  #i is the day, goes to 24 to test on 25 and end. :P
                for j in range(0,24): # j is the hour
                    print 'Beginning   day = ', i, '  hour =  ', j
                    num_round = 500
                    eta = 0.1
                    alpha = 0
                    ph0 = str(j).rjust(2,'0')  #the hour on which to train and test
                    p0 = str(i).rjust(2,'0')  #the day to train
                    p1 = str(i+1).rjust(2,'0')  #the day to test
                    #train_data, train_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p0+".npy")
                    train_data, train_label = format_data('/media/54D42AE2D42AC658/DataHourly/output_new_06'+p0+ph0+'.npy')
                    dtrain = xgb.DMatrix(train_data, label=train_label)
                    #test_data, test_label = format_data("/home/kbhalla/Desktop/Data/day_samp-06-"+p1+".npy")
                    test_data, test_label = format_data('/media/54D42AE2D42AC658/DataHourly/output_new_06'+p1+ph0+'.npy')
                    dtest = xgb.DMatrix(test_data, label=test_label)
                    evallist = [(dtrain,'train'), (dtest,'eval')]

                    modName = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2.model'
                    dumpname = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2.txt'
                    bst = getBst(dtrain, evallist, train_label, modName,num_round, eta,dumpname)

                    # train model 1A and 1B
                    # first, divide train in trainA (yhat = 0) and trainB  (yhat = 1) and
                    print 'predicting on dtrain...'
                    y_hatTrain = bst.predict(dtrain)
                    A, yA, B, yB = getAyAByB(y_hatTrain,train_data,train_label)

                    dtrainA = xgb.DMatrix(A, label=yA)
                    dtrainB = xgb.DMatrix(B, label=yB)
                    evallistA = [(dtrainA,'train'), (dtest,'eval')]
                    evallistB = [(dtrainB,'train'), (dtest,'eval')]
                    #### Create new models 1A and 1B
                    modNameA = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2_A.model'
                    dumpnameA = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2_A.txt'
                    modNameB = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2_B.model'
                    dumpnameB = '/home/rmendoza/Desktop/XGBoost/testHourly/testHourly' + p0 + '_to_' + p1 + ph0 + '_v2_B.txt'
                    bstA = getBst(dtrainA, evallistA, yA, modNameA,num_round, eta,dumpnameA)
                    bstB = getBst(dtrainB, evallistB, yB, modNameB,num_round, eta,dumpnameB)
                    #print 'B[0:20,-10:] = ',B[0:20,-10:]
                    # print 'yB[1:20] = '
                    # print yB[0:20]
                    # print 'y = y_pred > cut' =

                    ######### Predict/test the model on next day
                    y_true = test_label
                    y_pred = bst.predict(dtest)
                    ### Get the Xa and Xb for the predicted ones and zeroes
                    testA, testyA, testB, testyB = getAyAByBtest(y_pred,test_data,test_label)
                    #### Now simply predict on testA and testB with models 1A and 1B and get the numbers...
                    dtestA = xgb.DMatrix(testA, label=testyA)
                    y_predA = bstA.predict(dtestA)
                    dtestB = xgb.DMatrix(testB, label=testyB)
                    y_predB = bstB.predict(dtestB)
                    #print 'y_predB', y_predB
                    ### Getting the splits of negatives and positives for each one
                    testAneg, testyAneg, testApos, testyApos = getAyAByBtest(y_predA,testA,testyA)
                    testBneg, testyBneg, testBpos, testyBpos = getAyAByBtest(y_predB,testB,testyB)
                    ### Get the recall and Net Savings
                    labels = []
                    preds = []
                    labels.extend(testyAneg.tolist())
                    preds.extend([0]*len(testyAneg))
                    labels.extend(testyApos.tolist())
                    preds.extend([1]*len(testyApos))
                    labels.extend(testyBneg.tolist())
                    preds.extend([0]*len(testyBneg))
                    labels.extend(testyBpos.tolist())
                    preds.extend([1]*len(testyBpos))
                    labels = np.array(labels)
                    preds = np.array(preds)
                    recall = metrics.recall_score(labels, preds)
                    filter_rate = sum(np.logical_not(preds))/float(len(preds))
                    savings = 127000*filter_rate -5200 -850000*(1-recall)
                    #['Net_Savings', 'num_round', 'day_trained', 'day_predicted','hour_trainedAndTested']
                    results = [savings, num_round,p0, p1,ph0]
                    wr.writerow(results)
                    print 'done for the hour', j
                    print '--------------------------'
                print 'done for the DAY', i
                print '-------------------------------------'
                print '-------------------------------------'
            print '_______________________________________________________________________'
            print '_______________________________________________________________________'
        except Exception as e:
            print e
            print 'ooops'
            #pass
            errorCounter += 1
            print 'There was an error, count ', errorCounter
            subjeto = 'Error on code... countOfError' + str(errorCounter)
            #sendEmail('moralesmendozar@gmail.com',subjeto,"XGBoost-trainHoursDaily.py encountered an error. :P")
            #time.sleep(20)  #sleep


#sendEmail('moralesmendozar@gmail.com','Code Done2',"XGBoost-trainHoursDaily.py ended running in the local RIPS computer. :P")