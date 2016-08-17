import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import linear_model, preprocessing
try:
    import cPickle as pickle
except:
    import pickle

def GetData(month, day): ## Input Weiyi-formatted Data
    root = "/mnt/rips2/2016"
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    data = os.path.join(root,p0,p1,"day_samp_newer_bin.npy")
    print "Reading Data..."
    temp = np.load(data)
    Data = csr_matrix(( temp['data'], temp['indices'], temp['indptr']),
                         shape = temp['shape'], dtype=float).toarray()
    print "Finished reading data file"
    return Data


def DataFormat(month, day):
    Data = GetData(month, day)
    n = int(np.size(Data,0))
    print "Creating X,y"
    X = Data[:,:-1]
    X = BestK.transform(X)
    y = 2*Data[:,-1]-np.ones(n)
    print "X,y created"
    m = int(np.size(X,1))
    print "The number of data points is %s, each with %s features" % (n, m)
    print "Formatting ..."
    X_scaled = scaler.transform(X)
    K = np.count_nonzero(y+np.ones(n)) #Number of good data points
    print "Training on %s data points" % np.size(y)
    print "Formatted"
    return X_scaled, y, n, K# Training set plus some numbers useful for weighting




def TrainingModel(month, day):
    """
    Training the model on data
    :param month:
    :param day:
    :return: A trained model for Predictor (the filter) to use
    """
    X_train, y_train, n, K = DataFormat(month, day)
    clf = linear_model.SGDClassifier(learning_rate='optimal',
                             class_weight={-1:1, 1:2.5*n/K}, n_jobs=-1,alpha=0.000001, warm_start=True,
                             n_iter = 10, penalty='l2', average=True, eta0=0.0625)

    print "Training ... Warm Start = %s" % (True)
    clf.fit(X_train,y_train)
    root = '/mnt/rips2/2016'
    p0 = "0" + str(month)
    p1 = str(day).rjust(2,'0')
    pickle.dump(clf,open( os.path.join(root,p0,p1,"Models/SGD.p"), "wb"))


Data , label = GetData(6,19)[:,:-1], GetData(6, 19)[:,-1]
BestK = SelectKBest(chi2, k = 750)
BestK.fit(Data, label)
Data = BestK.transform(Data)
scaler = preprocessing.StandardScaler().fit(Data)
for month in range(6,7):
    for day in range(19,32):
        TrainingModel(month, day)
        # except:
        #     pass


