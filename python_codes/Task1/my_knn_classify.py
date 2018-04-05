import numpy as np
from scipy import stats

def my_knn_classify(Xtrn, Ctrn, Xtst, Ks):
    # Input:
    #   Xtrn : M-by-D ndarray of training data (dtype=np.float_)
    #   Ctrn : M-by-1 ndarray of labels for Xtrn (dtype=np.int_)
    #   Xtst : N-by-D ndarray of test data (dtype=np.float_)
    #   Ks   : List of the numbers of nearest neighbours in Xtrn
    # Output:
    #  Cpreds : N-by-L ndarray of predicted labels for Xtst (dtype=np.int_)
   # return
    N,D =np.shape(Xtst)
    M,D =np.shape(Xtrn)
    Cpreds = np.zeros((N,len(Ks)))
    maxk=np.amax(Ks)
    DI=distance(Xtst,Xtrn)

    firstk= DI[:, :maxk]
    labelling(Ctrn,firstk,maxk,N)
    for i in range(len(Ks)):
        Cpreds[i]=kclass(firstk,Ks[i])



    return Cpreds
def labelling(Ctrn,firstk,maxk,N):
    for i in range(N):
        for j in range(maxk):
            firstk[i][j]=Ctrn[firstk[i][j]]

def kclass(firstk,k):
    array= firstk[:, :k]
    classified=stats.mode(array,axis=1)
    return classified[0][0]



def distance(test, train):
    xx=np.sum(test*test, axis=1,dtype=np.float64, keepdims=True)
    yy=np.sum(train*train, axis=1,dtype=np.float64, keepdims=True)
    dist=np.dot(xx,yy.T)
    DI=xx-2*dist+yy.T
    sortdist=DI.argsort()
    return sortdist







