#! /usr/bin/env python
# A sample template for my_knn_system.py
import time;
import numpy as np
import scipy.io
from my_knn_classify import *
# import my_knn_classify as knnc

from my_confusion import *

# Load the data set
#   NB: replace <UUN> with your actual UUN.
filename = "/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1545545/data.mat";
data = scipy.io.loadmat(filename);

# Feature vectors: Convert uint8 to double, and divide by 255.
Xtrn = data['dataset']['train'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
Xtst = data['dataset']['test'][0,0]['images'][0,0].astype(dtype=np.float_) /255.0
# Labels : convert float64 to integer, and subtract 1 so that class number starts at 0 rather than 1.
Ctrn = data['dataset']['train'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctrn = Ctrn.reshape((Ctrn.size, 1))
Ctst = data['dataset']['test'][0,0]['labels'][0,0].astype(dtype=np.int_).flatten()-1
Ctst = Ctst.reshape((Ctst.size, 1))

#YourCode - Prepare measuring time

# Run K-NN classification
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)

#YourCode - Measure the user time taken, and display it.
start=time.clock()
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb)
end=time.clock()
print("The time taken is ", end-start, " seconds")

#YourCode - Get confusion matrix and accuracy for each k in kb.
for i in range(len(kb)):
    CM, ACC = my_confusion(Ctst, Cpreds[:i])


    print 'k value:', kb[i]
    print 'N value:', np.shape(Cpreds)[0]
    print 'Nerrs value:', ACC*np.shape(Cpreds)[0]
    print 'acc value:', ACC
#YourCode - Save each confusion matrix.






#YourCode - Display the required information - k, N, Nerrs, acc for each element of kb
