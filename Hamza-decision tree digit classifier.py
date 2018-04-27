# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:14:17 2018

@author: csr20
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data_train = pd.read_csv("C:/Users/Subiyyal/Desktop/Machine Learning/digits/train.csv").as_matrix()
data_test = pd.read_csv("C:/Users/Subiyyal/Desktop/Machine Learning/digits/test.csv").as_matrix()
#print(data)
clf = DecisionTreeClassifier()

#training dataset
xtrain = data_train[0:42000,1:]
train_label = data_train[0:42000, 0]

clf.fit(xtrain, train_label)

#testing data
xtest = data_test[0:28000, 0:]
actual_label = data_train[0:42000, 0]

#
d = xtest[20]
d.shape = (28,28)
pt.imshow(255-d, cmap = 'gray')
print(clf.predict([xtest[20]]))
pt.show

#p = clf.predict(xtest)
#
#count = 0
#for i in range(0,28000):
#    count+=1 if p[i] == actual_label[i] else 0
##print("accuracy = ", (count/28000)*100)
#
#print("accuracy = ", count)