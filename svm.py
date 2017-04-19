#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:23:43 2017

@author: yuhan
"""
import os
import glob
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn import model_selection
pretrained = np.loadtxt("pretrained_output.txt")
label = np.loadtxt("onehot_label.txt")

#data = np.concatenate((pretrained, label), axis=1)

def get_path_label(pstr):
    trainpaths = []
    paths = []
    labels = list()
    testpaths = []
    test_label = list()
    train_label = list()
    label_names = []
    dir1 = os.path.dirname(__file__)
    rpath = pstr
    i = 0
    j = 1
    filename = os.path.join(dir1, rpath)
    for root, dirs, files in os.walk(filename, topdown=False):
        for name in dirs:
            subdir = os.path.join(filename,name)
            breed = j
            label_names.append(name)
            for pic in os.listdir(subdir):
                if pic.endswith(".jpg"):
                    paths.append(os.path.join(subdir,pic))
                    labels.append(breed)
                    if i % 10 == 0:
                        testpaths.append(os.path.join(subdir,pic))
                        test_label.append(breed)
                    else:
                        trainpaths.append(os.path.join(subdir,pic))
                        train_label.append(breed)
                    i = i+1
            j = j +1
    return label_names

label_names = get_path_label("./new_data")
r,c = label.shape
y_train = np.zeros((r,1),dtype = np.int32)
y_names = []
for i in xrange(r):
    idarr = np.nonzero(label[i,] == 1)
    idx = np.take(idarr,0)
    y_train[i] = np.take(idx,0) + 1
    y_names.append(label_names[idx])
    
#print np.unique(y_train)

data = np.concatenate((pretrained, y_train), axis=1)
random.shuffle(data)
training_data = data[:1260,:1000]
training_label = data[:1260,1000:]
test_data = data[1260:,:1000]
test_label = data[1260:,1000:]

yy = np.ravel(y)

clf = tree.DecisionTreeClassifier()
clf2 = GaussianNB()
clf3 = svm.SVC(C=25, kernel='rbf', degree=3, gamma=15, coef0=4, shrinking=True, 
               probability=True, tol=0.005, cache_size=200, class_weight=None, 
               verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
#clf = clf.fit(training_data, training_label)
#clf2 = clf2.fit(training_data, training_label)

clf3 = clf3.fit(training_data, training_label) 

r1, c1 = test_label.shape
k1 = 0
k2 = 0
k3 = 0
for i in xrange(r1):
    #print np.take(np.nonzero(clf.predict(test_data[i,:].reshape(1,-1)) == 1),1)
    #print np.take(np.nonzero(test_label[i,] == 1),0)

    if( np.take(clf.predict(test_data[i,:].reshape(1,-1)),0) == np.take(test_label[i,],0) ):
        k1 = k1 +1
    
    if( np.take(clf2.predict(test_data[i,:].reshape(1,-1)),0) == np.take(test_label[i,],0) ):
        k2 = k2 +1
    
    if( np.take(clf3.predict(test_data[i,:].reshape(1,-1)),0) == np.take(test_label[i,],0) ):
        k3 = k3 +1
        #print np.take(clf3.predict(test_data[i,:].reshape(1,-1)),0)
print k3