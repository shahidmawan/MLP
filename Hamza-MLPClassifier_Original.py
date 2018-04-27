# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:14:17 2018

@author: csr20
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("C:/Users/Subiyyal/Desktop/Machine Learning/digits/train.csv").as_matrix()
#print(data)
clf = MLPClassifier()

#training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000, 0]

clf.fit(xtrain, train_label)

#testing dat
xtest = data[21000:, 1:]
actual_label = data[21000:, 0]

#d = xtest[1]
#d.shape = (28,28)
#pt.imshow(255-d, cmap = 'gray')
#print(clf.predict([xtest[1]]))
#pt.show

p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    count+=1 if p[i] == actual_label[i] else 0
print("accuracy = ", (count/21000)*100)